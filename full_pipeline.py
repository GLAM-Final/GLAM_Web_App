import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

import librosa
import numpy as np
import torch
from reasoning_pipeline import aggregate_reasoning_summaries
from interface import separate_audio_file, load_wav2vec

DEFAULT_INPUT_DIR = Path("input_mixes")
DEFAULT_OUTPUT_DIR = Path("separated_audios")
DEFAULT_RUNS_DIR = Path("gnn_runs")
DEFAULT_PIPELINE_OUTPUT_DIR = Path("pipeline_results")
DEFAULT_GNN_CHECKPOINT = "best_audio_separation_model.pt"
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac"]
FRAME_SECONDS = 0.5
SR = 16000
WINDOW_SECONDS = 5.0


def find_audio_files(directory: Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(sorted(directory.glob(f"*{ext}")))
    return sorted(files)


def infer_on_audio_file(
    audio_path: Path,
    gnn_model,
    processor,
    wav2vec_model,
    run_root: Path,
    device: torch.device,
    window_seconds: float = WINDOW_SECONDS,
    frame_seconds: float = FRAME_SECONDS,
) -> Optional[Dict]:
    import importlib

    gnn = importlib.import_module("gnn")
    build_chain_edge_index = getattr(gnn, "build_chain_edge_index")
    PatientStateManager = getattr(gnn, "PatientStateManager")
    torch_geometric_data = importlib.import_module("torch_geometric.data")
    Data = getattr(torch_geometric_data, "Data")

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    audio_duration_s = float(len(y) / SR)
    frame_len = int(frame_seconds * SR)

    frames = []
    for i in range(0, len(y), frame_len):
        f = y[i:i + frame_len]
        if len(f) < frame_len:
            f = np.pad(f, (0, frame_len - len(f)), mode="constant")
        frames.append(f.astype(np.float32))

    if not frames:
        return None

    embeddings = []
    batch_size = 8
    with torch.no_grad():
        for index in range(0, len(frames), batch_size):
            batch = frames[index:index + batch_size]
            inputs = processor(batch, sampling_rate=SR, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            out = wav2vec_model(input_values)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            embeddings.append(emb)

    X = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768), dtype=np.float32)
    if X.shape[0] == 0:
        return None

    edge_index = build_chain_edge_index(X.shape[0])
    if X.shape[0] == 0:
        return None

    edge_index = build_chain_edge_index(X.shape[0])
    data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index)

    start = time.time()
    with torch.no_grad():
        data = data.to(device)
        w_logits, c_logits = gnn_model(data)
    infer_ms = (time.time() - start) * 1000.0

    w_prob = float(torch.sigmoid(w_logits).view(-1)[0].cpu().item())
    c_prob = float(torch.sigmoid(c_logits).view(-1)[0].cpu().item())
    w_pred = int(w_prob >= 0.5)
    c_pred = int(c_prob >= 0.5)

    w_unc = c_unc = w_conf = c_conf = None
    if hasattr(gnn_model, "log_var_wheeze"):
        w_unc = float(np.exp(0.5 * float(gnn_model.log_var_wheeze.detach().cpu().item())))
        w_conf = float(1.0 / (1.0 + w_unc))
    if hasattr(gnn_model, "log_var_crackle"):
        c_unc = float(np.exp(0.5 * float(gnn_model.log_var_crackle.detach().cpu().item())))
        c_conf = float(1.0 / (1.0 + c_unc))

    breathing_rate_bpm = None
    if len(y) >= SR:
        breathing_rate_bpm = float(np.nan) if len(y) == 0 else None
        from gnn import estimate_breathing_rate_bpm

        breathing_rate_bpm = estimate_breathing_rate_bpm(y, SR, audio_duration_s)

    state_mgr = PatientStateManager(
        ema_alpha=0.12,
        low_delta=0.08,
        high_delta=0.20,
        min_samples_for_baseline=5,
        force_established_after_s=10.0,
    )

    frames_per_window = max(1, int(round(window_seconds / frame_seconds)))
    window_rows = []
    for w_start in range(0, X.shape[0], frames_per_window):
        Xw = X[w_start:w_start + frames_per_window]
        if Xw.shape[0] == 0:
            continue
        ew = build_chain_edge_index(Xw.shape[0])
        dw = Data(x=torch.tensor(Xw, dtype=torch.float32), edge_index=ew)

        with torch.no_grad():
            dw = dw.to(device)
            w_l, c_l = gnn_model(dw)

        w_p = float(torch.sigmoid(w_l).view(-1)[0].cpu().item())
        c_p = float(torch.sigmoid(c_l).view(-1)[0].cpu().item())
        w_pd = int(w_p >= 0.5)
        c_pd = int(c_p >= 0.5)

        start_sec = float(w_start * frame_seconds)
        end_sec = float(min((w_start + Xw.shape[0]) * frame_seconds, audio_duration_s))
        start_sample = int(start_sec * SR)
        end_sample = int(end_sec * SR)
        y_window = y[start_sample:end_sample] if end_sample > start_sample else np.array([], dtype=np.float32)
        breathing_rate_window = None
        if len(y_window) >= SR:
            from gnn import estimate_breathing_rate_bpm

            breathing_rate_window = estimate_breathing_rate_bpm(y_window, SR, max(end_sec - start_sec, 1e-6))

        state_out = state_mgr.update_and_get_state(audio_path.stem, w_p, c_p, timestamp=start_sec)

        window_rows.append({
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "num_frames": int(Xw.shape[0]),
            "wheeze_prob": round(w_p, 4),
            "crackle_prob": round(c_p, 4),
            "wheeze_pred": w_pd,
            "crackle_pred": c_pd,
            "patient_state": state_out.get("overall_state"),
            "breathing_rate_bpm": None if breathing_rate_window is None else round(breathing_rate_window, 2),
            "ig_topk": [],
            "gxi_topk": [],
        })

    run_id = time.strftime("%Y%m%dT%H%M%SZ") + "_" + audio_path.stem
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "request_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": {
            "audio_id": audio_path.stem,
            "audio_duration_s": round(audio_duration_s, 3),
            "wheeze": {
                "probability": round(w_prob, 4),
                "prediction": bool(w_pred),
                "confidence": None if w_conf is None else round(w_conf, 4),
            },
            "crackle": {
                "probability": round(c_prob, 4),
                "prediction": bool(c_pred),
                "confidence": None if c_conf is None else round(c_conf, 4),
            },
            "breathing_rate_bpm": None if breathing_rate_bpm is None else round(breathing_rate_bpm, 2),
            "model_version": str(gnn_model.__class__.__name__),
            "inference_time_ms": round(float(infer_ms), 2),
        },
        "reasoning": {
            "thresholds": {"wheeze": 0.5, "crackle": 0.5},
            "uncertainty_std": {"wheeze": w_unc, "crackle": c_unc},
            "flags": {"short_audio": bool(audio_duration_s < frame_seconds), "near_threshold": bool(abs(w_prob - 0.5) <= 0.02 or abs(c_prob - 0.5) <= 0.02)},
            "cumulative_windows": {
                "window_seconds": float(window_seconds),
                "frame_seconds": float(frame_seconds),
                "num_windows": int(len(window_rows)),
                "frames_per_window": int(frames_per_window),
                "wheeze_window_positive": int(sum(r["wheeze_pred"] for r in window_rows)),
                "crackle_window_positive": int(sum(r["crackle_pred"] for r in window_rows)),
                "wheeze_window_ratio": float(np.mean([r["wheeze_pred"] for r in window_rows])) if window_rows else 0.0,
                "crackle_window_ratio": float(np.mean([r["crackle_pred"] for r in window_rows])) if window_rows else 0.0,
                "wheeze_prob_mean_window": float(np.mean([r["wheeze_prob"] for r in window_rows])) if window_rows else 0.0,
                "crackle_prob_mean_window": float(np.mean([r["crackle_prob"] for r in window_rows])) if window_rows else 0.0,
            },
        },
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(run_dir / "reasoning.json", "w", encoding="utf-8") as f:
        json.dump(output["reasoning"], f, indent=2)
    with open(run_dir / "window_report.json", "w", encoding="utf-8") as f:
        json.dump({"audio_id": audio_path.stem, "windows": window_rows}, f, indent=2)
    with open(run_dir / "window_report.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "start_sec",
            "end_sec",
            "num_frames",
            "wheeze_prob",
            "crackle_prob",
            "wheeze_pred",
            "crackle_pred",
            "patient_state",
            "breathing_rate_bpm",
            "ig_topk",
            "gxi_topk",
        ])
        writer.writeheader()
        for row in window_rows:
            row_copy = dict(row)
            row_copy["ig_topk"] = ";".join(map(str, row_copy["ig_topk"]))
            row_copy["gxi_topk"] = ";".join(map(str, row_copy["gxi_topk"]))
            writer.writerow(row_copy)

    output["artifacts"] = {
        "result_json": "result.json",
        "reasoning_json": "reasoning.json",
        "window_report_json": "window_report.json",
        "window_report_csv": "window_report.csv",
        "run_dir": str(run_dir),
    }
    return output


def run_full_pipeline(
    input_dir: Path,
    sep_output_dir: Path,
    run_root: Path,
    pipeline_output_dir: Path,
    gnn_checkpoint: str,
    patient_names: Optional[List[str]] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    device = torch.device(device_str)
    sep_output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    mixture_files = find_audio_files(input_dir)
    if not mixture_files:
        raise FileNotFoundError(f"No mixture files found in {input_dir}")

    for mix_file in mixture_files:
        print(f"Separating mixture: {mix_file.name}")
        separate_audio_file(str(mix_file), sep_output_dir, patient_names=patient_names)

    print("Loading GNN and Wav2Vec models...")
    processor, wav2vec_model = load_wav2vec(device)
    from gnn import load_gnn_model

    gnn_model = load_gnn_model(gnn_checkpoint, device)

    separated_files = find_audio_files(sep_output_dir)
    if not separated_files:
        raise FileNotFoundError(f"No separated audio files found in {sep_output_dir}")

    behavior_results = []
    for audio_file in separated_files:
        print(f"Running GNN inference: {audio_file.name}")
        result = infer_on_audio_file(
            audio_file,
            gnn_model,
            processor,
            wav2vec_model,
            run_root,
            device,
            window_seconds=WINDOW_SECONDS,
            frame_seconds=FRAME_SECONDS,
        )
        if result is not None:
            behavior_results.append(result)

    output_path = pipeline_output_dir / "full_pipeline_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(behavior_results, f, indent=2)

    print("Aggregating reasoning summaries...")
    run_dirs = [Path(item["artifacts"]["run_dir"]) for item in behavior_results if item.get("artifacts")]
    aggregate_reasoning_summaries(run_dirs, pipeline_output_dir)

    print(f"Saved full pipeline results to: {output_path}")
    print(f"Saved reasoning summary to: {pipeline_output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full auditory separation + GNN reasoning pipeline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder of mixture audio files.")
    parser.add_argument("--sep-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for separated source audio.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS_DIR, help="Base folder for GNN run artifacts.")
    parser.add_argument("--pipeline-output-dir", type=Path, default=DEFAULT_PIPELINE_OUTPUT_DIR, help="Folder for pipeline summaries.")
    parser.add_argument("--gnn-checkpoint", type=str, default=DEFAULT_GNN_CHECKPOINT, help="Path to the GNN checkpoint file.")
    parser.add_argument("--patient-names", nargs="*", default=None, help="Optional names for separated sources.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patient_names = args.patient_names[:3] if args.patient_names else None
    run_full_pipeline(
        input_dir=args.input_dir,
        sep_output_dir=args.sep_output_dir,
        run_root=args.run_root,
        pipeline_output_dir=args.pipeline_output_dir,
        gnn_checkpoint=args.gnn_checkpoint,
        patient_names=patient_names,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
