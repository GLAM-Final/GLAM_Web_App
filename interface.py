import json
import os
import sqlite3
import shutil
import time
import torch
import torchaudio
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from itertools import permutations
from typing import Dict, List, Optional

from reasoning_pipeline import aggregate_reasoning_summaries
from model import load_model

MODEL_PATH = "best_audio_separation_model.pt"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def apply_wiener_filter(est_sources, mixture, iterations=1):
    """
    A simplified Wiener filter refinement. 
    In source separation, this often refers to re-masking the mixture 
    based on the relative energy of the estimated sources.
    """
    # eps to avoid division by zero
    eps = 1e-10
    # Square the estimates to get power/variance proxies
    est_power = np.maximum(np.abs(est_sources)**2, eps) 
    total_power = np.sum(est_power, axis=0, keepdims=True) + eps
    
    # The Wiener gain is est_power / total_power
    # We apply this gain to the original mixture
    refined_sources = (est_power / total_power) * mixture
    return refined_sources


def save_waveform_plot(waveform, sample_rate, title, path=None):
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    data = waveform.cpu().numpy()
    times = np.arange(data.shape[-1]) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(times, data, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
    ax.grid(True)
    fig.tight_layout()
    if path is None:
        temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(temp_image.name)
        plt.close(fig)
        return temp_image.name
    out_path = str(path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def predict_sources(waveform, sr, reference_waveforms=None):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            16000,
        )

    waveform = waveform.squeeze(0)
    with torch.no_grad():
        mixture = waveform.unsqueeze(0)
        prediction = get_model()(mixture)
        prediction = prediction.squeeze(0).permute(1, 0)
        
    # Convert to numpy for post-processing as per your evaluation snippet
    pred_np = prediction.cpu().numpy() # [3, T]
    mix_np = waveform.cpu().numpy()    # [T]

    # 1. Apply Wiener Filter
    refined_preds = apply_wiener_filter(pred_np, mix_np)

    # 2. Automated Permutation Alignment using Reference Audios
    if reference_waveforms is not None and len(reference_waveforms) > 0:
        best_perm = None
        min_mse = float('inf')
        num_sources = refined_preds.shape[0]

        for perm in permutations(range(num_sources)):
            current_mse = 0
            for i, p in enumerate(perm):
                if i < len(reference_waveforms) and reference_waveforms[i] is not None:
                    # Compare the first N samples to find the best match
                    ref = reference_waveforms[i]
                    length = min(refined_preds[p].shape[0], ref.shape[0])
                    current_mse += np.mean((ref[:length] - refined_preds[p][:length]) ** 2)
            
            if current_mse < min_mse:
                min_mse = current_mse
                best_perm = perm
        
        refined_preds = np.array([refined_preds[p] for p in best_perm])

    return torch.from_numpy(refined_preds), 16000


def save_separated_sources(prediction, output_dir, base_name, patient_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_names = patient_names or ["Patient 1", "Patient 2", "Patient 3"]
    patient_names = [name if name else f"Patient {idx + 1}" for idx, name in enumerate(patient_names)]

    output_audio_paths = []
    output_image_paths = []

    for i in range(prediction.shape[0]):
        audio_name = f"{base_name}_s{i+1}.wav"
        audio_path = output_dir / audio_name
        save_audio_file(
            str(audio_path),
            prediction[i].unsqueeze(0),
            16000,
        )
        output_audio_paths.append(str(audio_path))

        image_name = f"{base_name}_s{i+1}_waveform.png"
        image_path = output_dir / image_name
        save_waveform_plot(
            prediction[i],
            16000,
            title=f"Separated waveform - {patient_names[i]}",
            path=image_path,
        )
        output_image_paths.append(str(image_path))

    return output_audio_paths, output_image_paths


def separate_audio_file(audio_path, output_dir, patient_names=None):
    waveform, sr = load_audio(audio_path)
    prediction, sample_rate = predict_sources(waveform, sr)
    return save_separated_sources(prediction, output_dir, Path(audio_path).stem, patient_names)


def load_audio(audio_path):
    try:
        return torchaudio.load(audio_path)
    except (ImportError, RuntimeError, OSError):
        try:
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
            return waveform, sr
        except (RuntimeError, ValueError, OSError):
            data, sr = librosa.load(audio_path, sr=None, mono=False)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data)
            return waveform, sr


def save_audio_file(path, waveform, sample_rate):
    audio = waveform.detach().cpu().numpy()
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] > 1:
        audio = audio.T
    else:
        audio = audio[0]
    sf.write(path, audio, sample_rate) # save_audio_file now only saves the audio


RESULTS_DIR = Path("pipeline_results")
REASONING_SUMMARY_PATH = RESULTS_DIR / "reasoning_summary.json"
PATIENT_REGISTRY_PATH = RESULTS_DIR / "patient_registry.json"
HISTORY_RECORDS_PATH = RESULTS_DIR / "history_records.json"
DB_PATH = RESULTS_DIR / "speformer.db"
AUDIO_STORAGE_DIR = RESULTS_DIR / "audio"
GNN_RUN_ROOT = RESULTS_DIR / "gnn_runs"
DEFAULT_GNN_CHECKPOINT = "best_audio_separation_model.pt"


def _get_db_connection():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def initialize_database():
    conn = _get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_registry (
            id INTEGER PRIMARY KEY,
            patient_id TEXT UNIQUE,
            name TEXT,
            reference_audio TEXT,
            local_reference_audio TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history_records (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            mix_audio TEXT,
            local_mix_audio TEXT,
            patient_names TEXT,
            separated_sources TEXT,
            run_dirs TEXT,
            reasoning_summary_path TEXT,
            reasoning_count INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            file_type TEXT,
            patient_id TEXT,
            created_at TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _record_audio_file_metadata(path: str, file_type: str, patient_id: Optional[str] = None, notes: Optional[str] = None):
    try:
        initialize_database()
        conn = _get_db_connection()
        conn.execute(
            "INSERT OR IGNORE INTO audio_files (path, file_type, patient_id, created_at, notes) VALUES (?, ?, ?, ?, ?)",
            (str(Path(path).resolve()), file_type, patient_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), notes),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def _copy_audio_to_storage(src_path: Optional[str], subdir: str, prefix: str) -> Optional[str]:
    if not src_path:
        return None
    src = Path(src_path)
    try:
        if not src.exists():
            return None
        src_resolved = src.resolve()
        if RESULTS_DIR in src_resolved.parents or src_resolved == RESULTS_DIR.resolve():
            return str(src_resolved)
    except Exception:
        pass

    storage_dir = AUDIO_STORAGE_DIR / subdir
    storage_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    dest = storage_dir / f"{prefix}_{timestamp}_{src.name}"
    shutil.copy2(src, dest)
    _record_audio_file_metadata(str(dest), subdir, prefix)
    return str(dest)


def save_patient_registry(patient_entries: List[Dict[str, str]], registry_path: Optional[Path] = None) -> Path:
    path = Path(registry_path) if registry_path is not None else PATIENT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    initialize_database()
    conn = _get_db_connection()
    local_entries = []
    for entry in patient_entries:
        patient_id = entry.get("patient_id") or entry.get("id") or ""
        name = entry.get("name") or ""
        ref_audio = entry.get("reference_audio")
        local_ref_audio = _copy_audio_to_storage(ref_audio, "patient_reference", patient_id) if ref_audio else None
        conn.execute(
            "INSERT OR REPLACE INTO patient_registry (patient_id, name, reference_audio, local_reference_audio, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                patient_id,
                name,
                str(ref_audio) if ref_audio else None,
                local_ref_audio,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
        local_entries.append({
            "patient_id": patient_id,
            "name": name,
            "reference_audio": local_ref_audio or (str(ref_audio) if ref_audio else None),
        })
    conn.commit()
    conn.close()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(local_entries, f, indent=2)
    return path


def load_patient_registry(registry_path: Optional[Path] = None) -> List[Dict[str, str]]:
    if registry_path is None and DB_PATH.exists():
        try:
            initialize_database()
            conn = _get_db_connection()
            cursor = conn.execute("SELECT patient_id, name, local_reference_audio AS reference_audio FROM patient_registry ORDER BY id")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                return [{"patient_id": row["patient_id"], "name": row["name"], "reference_audio": row["reference_audio"]} for row in rows]
        except Exception:
            pass

    path = Path(registry_path) if registry_path is not None else PATIENT_REGISTRY_PATH
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def resolve_patient_names(patient_names=None, registry_path: Optional[Path] = None):
    registry = load_patient_registry(registry_path)
    resolved = []
    patient_names = patient_names or []
    for idx in range(3):
        name = None
        if idx < len(patient_names) and patient_names[idx]:
            name = patient_names[idx]
        elif idx < len(registry) and registry[idx].get("name"):
            name = registry[idx]["name"]
        if not name:
            name = f"Patient {idx + 1}"
        resolved.append(name)
    return resolved


def load_reasoning_summary(results_path: Optional[Path] = None):
    path = Path(results_path) if results_path is not None else REASONING_SUMMARY_PATH
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_history_records(history_path: Optional[Path] = None) -> List[Dict[str, str]]:
    if DB_PATH.exists():
        try:
            initialize_database()
            conn = _get_db_connection()
            cursor = conn.execute("SELECT * FROM history_records ORDER BY id")
            rows = cursor.fetchall()
            conn.close()
            history = []
            for row in rows:
                local_mix_audio = row["local_mix_audio"] or row["mix_audio"]
                history.append({
                    "timestamp": row["timestamp"],
                    "mix_audio": local_mix_audio,
                    "patient_names": json.loads(row["patient_names"] or "[]"),
                    "separated_sources": json.loads(row["separated_sources"] or "[]"),
                    "run_dirs": json.loads(row["run_dirs"] or "[]"),
                    "reasoning_summary_path": row["reasoning_summary_path"],
                    "reasoning_count": row["reasoning_count"],
                    "local_mix_audio": row["local_mix_audio"],
                })
            return history
        except Exception:
            pass
    path = Path(history_path) if history_path is not None else HISTORY_RECORDS_PATH
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def append_history_record(record: Dict[str, object], history_path: Optional[Path] = None) -> Path:
    path = Path(history_path) if history_path is not None else HISTORY_RECORDS_PATH
    local_mix_audio = _copy_audio_to_storage(record.get("mix_audio"), "mix_audio", "mix_audio") if record.get("mix_audio") else None
    if local_mix_audio:
        record["mix_audio"] = local_mix_audio

    initialize_database()
    conn = _get_db_connection()
    conn.execute(
        "INSERT INTO history_records (timestamp, mix_audio, local_mix_audio, patient_names, separated_sources, run_dirs, reasoning_summary_path, reasoning_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.get("timestamp"),
            str(record.get("mix_audio")) if record.get("mix_audio") else None,
            local_mix_audio,
            json.dumps(record.get("patient_names") or []),
            json.dumps(record.get("separated_sources") or []),
            json.dumps(record.get("run_dirs") or []),
            str(record.get("reasoning_summary_path")) if record.get("reasoning_summary_path") else None,
            int(record.get("reasoning_count") or 0),
        ),
    )
    conn.commit()
    conn.close()

    history = load_history_records(path)
    history.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def load_wav2vec(device: torch.device):
    import importlib

    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Required package 'transformers' is not installed. Please install it with `pip install transformers` and restart the app."
        ) from exc

    Wav2Vec2Model = getattr(transformers, "Wav2Vec2Model")
    Wav2Vec2Processor = getattr(transformers, "Wav2Vec2Processor")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model = model.to(device)
    model.eval()
    return processor, model


_live_sr = 16000
LIVE_PROCESSING_WINDOW_SECONDS = 5.0
LIVE_OVERLAP_SECONDS = 1.0

_live_processor = None
_live_wav2vec_model = None
_live_gnn_model = None
_live_device = None


def _initialize_models_for_live_processing():
    global _live_processor, _live_wav2vec_model, _live_gnn_model, _live_device
    if _live_device is None:
        _live_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _live_processor is None or _live_wav2vec_model is None:
        _live_processor, _live_wav2vec_model = load_wav2vec(_live_device)
    if _live_gnn_model is None:
        from gnn import load_gnn_model
        _live_gnn_model = load_gnn_model(DEFAULT_GNN_CHECKPOINT, _live_device)
    return _live_processor, _live_wav2vec_model, _live_gnn_model, _live_device


def process_audio_chunk_for_separation(audio_chunk_tensor):
    """Expects a 1D tensor at 16kHz."""
    prediction, _ = predict_sources(audio_chunk_tensor.unsqueeze(0), 16000)
    return prediction


def infer_on_separated_chunk(
    separated_chunk_np,
    gnn_model,
    processor,
    wav2vec_model,
    device,
    patient_manager,
    patient_id,
    timestamp,
):
    """Runs behavior inference and clinical state tracking on a single audio chunk."""
    inputs = processor(separated_chunk_np, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        out = wav2vec_model(input_values)
        emb = out.last_hidden_state.mean(dim=1)
        from gnn import build_chain_edge_index
        from torch_geometric.data import Data
        edge_index = build_chain_edge_index(1).to(device)
        data = Data(x=emb, edge_index=edge_index)
        w_logits, c_logits = gnn_model(data)
        w_prob = float(torch.sigmoid(w_logits).view(-1)[0].item())
        c_prob = float(torch.sigmoid(c_logits).view(-1)[0].item())

    from gnn import estimate_breathing_rate_bpm
    br_bpm = estimate_breathing_rate_bpm(separated_chunk_np, 16000, len(separated_chunk_np)/16000)

    return patient_manager.update_and_get_clinical_state(patient_id, w_prob, c_prob, breathing_rate=br_bpm, timestamp=timestamp)


def run_end_to_end(
    mix_audio_path,
    patient_names=None,
    reference_audio_paths=None, # New argument to accept reference audio paths
    gnn_checkpoint: Optional[str] = None,
    device_str: Optional[str] = None,
):
    if not mix_audio_path:
        raise ValueError("No mixture audio file provided.")

    # Note: reference_audio_paths are passed but not used by the current separate_audio/predict_sources
    patient_names = resolve_patient_names(patient_names)
    outputs = separate_audio(
        mix_audio_path,
        patient_names=patient_names,
        reference_audio_paths=reference_audio_paths, # Pass reference audio paths
    )
    separated_audio_paths = outputs[:3]

    device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    processor, wav2vec_model = load_wav2vec(device)

    from gnn import load_gnn_model

    gnn_model = load_gnn_model(gnn_checkpoint or DEFAULT_GNN_CHECKPOINT, device)

    GNN_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    behavior_results = []
    for audio_path in separated_audio_paths:
        from full_pipeline import infer_on_audio_file

        result = infer_on_audio_file(
            Path(audio_path),
            gnn_model,
            processor,
            wav2vec_model,
            GNN_RUN_ROOT,
            device,
        )
        if result is not None:
            behavior_results.append(result)

    run_dirs = [Path(item["artifacts"]["run_dir"]) for item in behavior_results if item.get("artifacts")]

    # Construct metadata mapping the audio_id (filename stem) to the selected patient name for human-readable reporting
    patient_meta = {}
    for i, audio_path in enumerate(separated_audio_paths):
        if audio_path:
            stem = Path(audio_path).stem
            if i < len(patient_names):
                patient_meta[stem] = {"name": patient_names[i]}

    reasoning_summaries = aggregate_reasoning_summaries(run_dirs, RESULTS_DIR, patient_meta=patient_meta)

    history_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mix_audio": str(mix_audio_path),
        "patient_names": patient_names,
        "separated_sources": separated_audio_paths,
        "run_dirs": [str(path) for path in run_dirs],
        "reasoning_summary_path": str(REASONING_SUMMARY_PATH),
        "reasoning_count": len(reasoning_summaries),
    }
    append_history_record(history_record)

    return outputs, reasoning_summaries, history_record


def monitoring_table_rows(results_path: Optional[Path] = None):
    records = load_reasoning_summary(results_path)
    rows = []
    for item in records:
        rows.append([
            item.get("audio_id"),
            item.get("overall_state"),
            item.get("mean_wheeze_prob"),
            item.get("mean_crackle_prob"),
            item.get("breathing_rate_mean"),
            item.get("comment"),
        ])
    return rows


def search_history_records(query: str, history_path: Optional[Path] = None):
    if not query:
        return []
    query_lower = query.strip().lower()
    records = load_history_records(history_path)
    results = []
    for item in records:
        patient_names = item.get("patient_names") or []
        mix_audio = str(item.get("mix_audio", ""))
        timestamp = str(item.get("timestamp", ""))
        if (
            query_lower in mix_audio.lower()
            or query_lower in timestamp.lower()
            or any(query_lower in str(name).lower() for name in patient_names)
        ):
            results.append([
                item.get("timestamp"),
                mix_audio,
                ", ".join([str(name) for name in patient_names if name]),
                item.get("reasoning_count"),
            ])
    return results


def search_reasoning_records(query: str, results_path: Optional[Path] = None):
    if not query:
        return []
    query_lower = query.strip().lower()
    records = load_reasoning_summary(results_path)
    results = []
    for item in records:
        audio_id = str(item.get("audio_id", ""))
        if query_lower in audio_id.lower():
            results.append([
                item.get("audio_id"),
                item.get("overall_state"),
                item.get("mean_wheeze_prob"),
                item.get("mean_crackle_prob"),
                item.get("breathing_rate_mean"),
                item.get("comment"),
            ])
    return results


def separate_audio(audio_path, patient_names=None, reference_audio_paths=None): # New argument

    waveform, sr = load_audio(audio_path)
    
    # Load reference waveforms if provided for alignment
    ref_waveforms = []
    if reference_audio_paths:
        for path in reference_audio_paths:
            if path and Path(path).exists():
                ref_wav, _ = load_audio(path)
                ref_waveforms.append(ref_wav.mean(dim=0).cpu().numpy())
            else:
                ref_waveforms.append(None)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            16000,
        )

    waveform = waveform.squeeze(0)

    with torch.no_grad():
        # Now passing the references to predict_sources for alignment logic
        prediction, _ = predict_sources(waveform.unsqueeze(0), 16000, reference_waveforms=ref_waveforms)

    patient_names = resolve_patient_names(patient_names)
    output_dir = AUDIO_STORAGE_DIR / "separated" / Path(audio_path).stem
    output_audio_paths, output_image_paths = save_separated_sources(
        prediction,
        output_dir,
        Path(audio_path).stem,
        patient_names=patient_names,
    )

    while len(output_audio_paths) < 3:
        output_audio_paths.append("")
    while len(output_image_paths) < 3:
        output_image_paths.append("")

    return output_audio_paths + output_image_paths