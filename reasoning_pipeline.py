import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def summarize_run_dir(run_dir: Path, default_window_seconds: float = 5.0, patient_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict]:
    win_json = run_dir / "window_report.json"
    win_csv = run_dir / "window_report.csv"
    data = None
    windows = []
    audio_id = run_dir.name

    if win_json.exists():
        try:
            data = json.loads(win_json.read_text(encoding="utf-8"))
            windows = data.get("windows", [])
            audio_id = data.get("audio_id") or run_dir.name
        except Exception:
            return None
    elif win_csv.exists():
        try:
            df = pd.read_csv(win_csv)
            windows = df.to_dict(orient="records")
            audio_id = df["audio_id"].iloc[0] if "audio_id" in df.columns else run_dir.name
        except Exception:
            return None
    else:
        return None

    if not windows:
        return None

    w_probs, c_probs, br_rates = [], [], []
    for w in windows:
        wp = safe_float(w.get("wheeze_prob"))
        cp = safe_float(w.get("crackle_prob"))
        if wp is not None:
            w_probs.append(wp)
        if cp is not None:
            c_probs.append(cp)
        br = safe_float(w.get("breathing_rate_bpm"))
        if br is not None:
            br_rates.append(br)

    w_mean = float(statistics.mean(w_probs)) if w_probs else None
    c_mean = float(statistics.mean(c_probs)) if c_probs else None
    br_mean = float(statistics.mean(br_rates)) if br_rates else None
    br_min = float(min(br_rates)) if br_rates else None
    br_max = float(max(br_rates)) if br_rates else None

    severity_rank = {"establishing": 0, "green": 1, "orange": 2, "red": 3}
    state_color_map = {"green": "green", "orange": "orange", "red": "red", "establishing": "grey"}

    meta = {}
    if isinstance(patient_meta, dict):
        meta = patient_meta.get(audio_id, {}) or {}
    age = float(meta.get("age", 40)) if isinstance(meta.get("age", None), (int, float)) else 40.0
    sex = str(meta.get("sex", "male"))

    def assess_rr(bpm: Optional[float]) -> Optional[Dict[str, Any]]:
        if bpm is None:
            return None
        try:
            from gnn import ClinicalReferenceRanges

            return ClinicalReferenceRanges.assess_respiratory_rate(float(bpm), age, sex)
        except Exception:
            return None

    processed_states = []
    latest_comment = ""

    for idx, w in enumerate(windows):
        pstate = str(w.get("patient_state") or "establishing").lower().strip()
        start_sec = safe_float(w.get("start_sec"))
        if start_sec is None:
            start_sec = float(idx * default_window_seconds)

        base_state = "establishing" if start_sec < 10.0 else (pstate if pstate in state_color_map else "establishing")
        rr_bpm = safe_float(w.get("breathing_rate_bpm"))
        rr_assessment = assess_rr(rr_bpm)
        rr_sev = rr_assessment.get("severity") if isinstance(rr_assessment, dict) else None

        final_state = base_state
        if rr_sev in ("orange", "red") and severity_rank.get(rr_sev, 0) > severity_rank.get(final_state, 0):
            final_state = rr_sev

        state_color = state_color_map.get(final_state, "grey")
        if final_state == "red":
            comment = "RED - patient requires clinical review."
        elif final_state == "orange":
            comment = "ORANGE - nurse should be cautious."
        elif final_state == "green":
            comment = "GREEN - no immediate attention required."
        else:
            comment = "GREY - establishing baseline; interpret with caution."
        if rr_sev in ("orange", "red"):
            comment += " Respiratory rate flagged."

        w["state_color"] = state_color
        w["comment"] = comment
        if rr_sev is not None:
            w["rr_severity"] = rr_sev
        if isinstance(rr_assessment, dict):
            w["rr_status"] = rr_assessment.get("status")

        processed_states.append(final_state)
        latest_comment = comment

    counts = Counter(processed_states)
    overall = next((s for s in ["red", "orange", "green", "establishing"] if counts.get(s)), "establishing")

    combined = latest_comment
    extras = []
    if w_mean is not None:
        extras.append(f"mean wheeze prob {w_mean:.3f}")
    if c_mean is not None:
        extras.append(f"mean crackle prob {c_mean:.3f}")
    if br_mean is not None:
        extras.append(f"mean breathing rate {br_mean:.1f} bpm (min {br_min:.1f}, max {br_max:.1f})")
    if extras:
        combined = f"{combined} {'; '.join(extras)}."

    try:
        if win_json.exists() and data is not None:
            data["windows"] = windows
            win_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        if win_csv.exists():
            df_out = pd.DataFrame(windows)
            for col in ["state_color", "comment", "rr_severity", "rr_status"]:
                if col not in df_out.columns:
                    df_out[col] = ""
            for col in ["ig_topk", "gxi_topk"]:
                if col in df_out.columns:
                    df_out[col] = df_out[col].apply(
                        lambda v: ";".join(str(x) for x in v) if isinstance(v, (list, tuple)) else ("" if pd.isna(v) else str(v))
                    )
            df_out.to_csv(win_csv, index=False)
    except Exception:
        pass

    return {
        "run_dir": str(run_dir),
        "audio_id": str(meta.get("name", audio_id)),
        "num_windows": int(len(windows)),
        "counts": dict(counts),
        "mean_wheeze_prob": w_mean,
        "mean_crackle_prob": c_mean,
        "breathing_rate_mean": br_mean,
        "breathing_rate_min": br_min,
        "breathing_rate_max": br_max,
        "overall_state": overall,
        "comment": latest_comment,
        "latest_state_color": windows[-1].get("state_color") if windows else None,
        "combined_reasoning": combined,
    }


def aggregate_reasoning_summaries(run_dirs: List[Path], output_dir: Path, patient_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reasoning_summaries = []
    for run_dir in run_dirs:
        summary = summarize_run_dir(run_dir, patient_meta=patient_meta)
        if summary:
            reasoning_summaries.append(summary)

    df = pd.DataFrame(reasoning_summaries)
    reasoning_json = output_dir / "reasoning_summary.json"
    reasoning_csv = output_dir / "reasoning_summary.csv"
    df.to_csv(reasoning_csv, index=False)
    with reasoning_json.open("w", encoding="utf-8") as f:
        json.dump(reasoning_summaries, f, indent=2)
    return reasoning_summaries


def load_reasoning_df(output_dir: Path) -> Optional[pd.DataFrame]:
    reasoning_json = output_dir / "reasoning_summary.json"
    if reasoning_json.exists():
        try:
            return pd.read_json(reasoning_json)
        except Exception:
            return None
    return None


def plot_patient_summary(df: pd.DataFrame, output_dir: Path, window_seconds: float = 5.0) -> None:
    if df is None or df.empty:
        print("No patient summaries to visualize")
        return

    audio_ids = df["audio_id"].fillna("unknown").astype(str).tolist()
    wheeze_mean = df["mean_wheeze_prob"].fillna(0).astype(float).tolist()
    crackle_mean = df["mean_crackle_prob"].fillna(0).astype(float).tolist()
    br_mean = df["breathing_rate_mean"].fillna(0).astype(float).tolist()

    x = np.arange(len(audio_ids))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width / 2, wheeze_mean, width, label=f"Mean Wheeze ({window_seconds}s)", color="coral")
    ax1.bar(x + width / 2, crackle_mean, width, label=f"Mean Crackle ({window_seconds}s)", color="skyblue")
    ax1.set_ylabel("Mean Probability")
    ax1.set_xlabel("Audio / Patient")
    ax1.set_xticks(x)
    ax1.set_xticklabels(audio_ids, rotation=45, ha="right")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Per-Patient 5s Window Means + Breathing Rate")

    ax2 = ax1.twinx()
    ax2.plot(x, br_mean, color="green", marker="o", label="Breathing Rate (bpm)")
    ax2.set_ylabel("Breathing Rate (bpm)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plot_path = output_dir / "patient_summary_5s.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Visualization saved to:", str(plot_path))
