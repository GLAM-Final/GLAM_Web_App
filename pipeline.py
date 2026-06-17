import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional

from interface import separate_audio_file

DEFAULT_INPUT_DIR = Path("input_mixes")
DEFAULT_OUTPUT_DIR = Path("separated_audios")
DEFAULT_PIPELINE_OUTPUT_DIR = Path("pipeline_results")
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac"]


def find_audio_files(directory: Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(sorted(directory.glob(f"*{ext}")))
    return sorted(audio_files)


def save_pipeline_summary(results: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "pipeline_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_csv = output_dir / "pipeline_summary.csv"
    fieldnames = [
        "input_file",
        "output_base",
        "num_sources",
        "audio_paths",
        "image_paths",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "input_file": item["input_file"],
                "output_base": item["output_base"],
                "num_sources": item["num_sources"],
                "audio_paths": ";".join(item["audio_paths"]),
                "image_paths": ";".join(item["image_paths"]),
            })


def process_separation_folder(
    input_dir: Path,
    output_dir: Path,
    pipeline_output_dir: Path,
    patient_names: Optional[List[str]] = None,
) -> List[Dict]:
    audio_files = find_audio_files(input_dir)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for audio_path in audio_files:
        print(f"Separating: {audio_path.name}")
        audio_paths, image_paths = separate_audio_file(
            str(audio_path),
            output_dir,
            patient_names=patient_names,
        )

        output_base = audio_path.stem
        results.append({
            "input_file": str(audio_path),
            "output_base": output_base,
            "num_sources": len(audio_paths),
            "audio_paths": audio_paths,
            "image_paths": image_paths,
        })

    save_pipeline_summary(results, pipeline_output_dir)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the separation pipeline over a folder of mixture audio files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing mixture audio files to separate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where separated sources and waveform images will be written.",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        default=DEFAULT_PIPELINE_OUTPUT_DIR,
        help="Folder where the pipeline summary files are written.",
    )
    parser.add_argument(
        "--patient-names",
        nargs="*",
        default=None,
        help="Optional patient names for naming separated outputs. Up to three names may be provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patient_names = None
    if args.patient_names:
        patient_names = args.patient_names[:3]

    results = process_separation_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pipeline_output_dir=args.pipeline_output_dir,
        patient_names=patient_names,
    )

    print(f"Processed {len(results)} files")
    print(f"Summary saved to: {args.pipeline_output_dir / 'pipeline_summary.json'}")
    print(f"CSV saved to: {args.pipeline_output_dir / 'pipeline_summary.csv'}")


if __name__ == "__main__":
    main()
