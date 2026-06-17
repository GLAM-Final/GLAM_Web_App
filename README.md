---
title: SpeFormer Respiratory Audio Pipeline
sdk: gradio
sdk_version: 5.16.0
python_version: 3.12
emoji: 🏢
colorFrom: blue
colorTo: indigo
pinned: false
short_description: Respiratory monitoring system
---

# SpeFormer Pipeline

This repository includes a respiratory audio separation pipeline built around a local SepFormer model and a clinical reasoning/GNN inference path.

## Key Files

### model.py

Loads the SepFormer separation model using local pretrained files in `pretrained_sepformer`.

Wraps encoder/masknet/decoder into `UnifiedSepFormer`.

### interface.py

Provides shared audio loading, separation, and saving helpers.

Includes:

* `separate_audio_file(...)`
* `predict_sources(...)`
* `save_separated_sources(...)`

for pipeline reuse.

Still supports the existing Gradio UI workflow via:

* `separate_audio(...)`

### pipeline.py

Runs separation over a directory of mixture audio files.

Writes separated WAVs and waveform PNGs to an output folder.

Produces:

* `pipeline_summary.json`
* `pipeline_summary.csv`

### gnn.py

Defines the GNN model architecture used for respiratory audio reasoning.

Provides utilities for:

* loading a GNN checkpoint
* estimating breathing rate
* patient state tracking
* clinical alert generation

### reasoning_pipeline.py

Processes GNN run outputs:

* `window_report.json`
* `window_report.csv`

Generates reasoning summary CSV/JSON files.

Includes visualization helpers for per-patient summary plots.

### full_pipeline.py

Runs the complete flow:

1. Separate mixture audio files
2. Run Wav2Vec2 + GNN inference
3. Save per-run artifacts
4. Aggregate reasoning summaries

## Usage

### Run Separation Only

```bash
python pipeline.py \
  --input-dir input_mixes \
  --output-dir separated_audios \
  --pipeline-output-dir pipeline_results
```

### Run Full Pipeline

```bash
python full_pipeline.py \
  --input-dir input_mixes \
  --sep-output-dir separated_audios \
  --run-root gnn_runs \
  --pipeline-output-dir pipeline_results \
  --gnn-checkpoint best_improved_30epochs_no_earlystop.pt
```

## Notes

* Place mixture audio files in `input_mixes` or provide a custom `--input-dir`.
* The separation model loads local checkpoint files from `pretrained_sepformer`.
* Full inference requires a valid GNN checkpoint file.
* Wav2Vec2 is used for respiratory behavior inference.
* Clinical reasoning summaries are written to `pipeline_results`.