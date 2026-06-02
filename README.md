# ESP32-CAM Paper Shadow Segmentation

Tiny real-time paper-shadow segmentation pipeline for ESP32-CAM class devices.

The project trains a low-resolution semantic segmentation model that predicts:

- `0`: paper/background
- `1`: paper shadow
- `2`: hand/object dark negative

The deployment target is a regular ESP32-CAM style board with OV2640 and PSRAM, using a `96x96` grayscale input and a `48x48` mask output.

## Project Layout

```text
src/shadowcam/          Training, model, metrics, preprocessing
scripts/                Dataset preparation, training, export, evaluation
esp32/                  TFLite Micro integration notes and runtime helpers
outputs/                User-facing artifacts
work/                   Local datasets, checkpoints, scratch files
```

## Install

Use a Python environment with the dependencies in `requirements.txt`.

```bash
python3 -m venv work/.venv
source work/.venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset Format

All training scripts use this unified layout:

```text
work/datasets/paper_shadow/
  train/
    images/*.png
    masks/*.png
  val/
    images/*.png
    masks/*.png
```

Mask pixel values:

```text
0 = paper/background
1 = paper shadow
2 = hand/object dark negative
```

Binary source masks with `0/255` are automatically converted to `0/1`.

## Prepare Open Datasets

Inspect a Hugging Face dataset before downloading the full content:

```bash
python scripts/inspect_hf_dataset.py \
  --dataset Donghyun99/ISTD \
  --cache-dir work/hf_cache \
  --timeout 90
```

Start with the Hugging Face ISTD mirror:

```bash
python scripts/prepare_istd_hf.py \
  --dataset Donghyun99/ISTD \
  --output work/datasets/istd_unified \
  --cache-dir work/hf_cache \
  --limit 1870
```

For a quick real-data pipeline check:

```bash
python scripts/prepare_istd_hf.py \
  --dataset Donghyun99/ISTD \
  --output work/datasets/istd_unified_sample \
  --cache-dir work/hf_cache \
  --limit 20
python scripts/synthesize_hard_negatives.py \
  --input work/datasets/istd_unified_sample \
  --output work/datasets/istd_hardneg_sample \
  --synthetic-objects 20 \
  --val-synthetic-objects 5
```

Convert a local SD7K folder after downloading it from the authors:

```bash
python scripts/prepare_sd7k_local.py \
  --source /path/to/SD7K \
  --output work/datasets/sd7k_unified
```

Add hard negatives from hand/object masks or synthetic dark shapes:

```bash
python scripts/synthesize_hard_negatives.py \
  --input work/datasets/istd_unified \
  --output work/datasets/paper_shadow \
  --synthetic-objects 1200
```

For a dependency and pipeline smoke test without downloading external data:

```bash
python scripts/generate_synthetic_paper_dataset.py \
  --output work/datasets/synthetic_paper_shadow \
  --train 160 \
  --val 40
```

Inspect any unified dataset before training:

```bash
python scripts/inspect_unified_dataset.py \
  --data work/datasets/paper_shadow
```

## Train

```bash
python scripts/train.py \
  --data work/datasets/paper_shadow \
  --out work/runs/tiny_shadow_v1 \
  --epochs 80 \
  --batch-size 64
```

### Modular Shadow + Exclusion Experiment

Use this route when hand/object false positives are the main problem. The model
shares one tiny backbone and emits two independent heads:

- `shadow_logits`: shadow probability.
- `exclude_logits`: hand/object/dark-negative probability.

The ESP32 mask is then:

```text
shadow = shadow_probability >= threshold AND exclude_probability < threshold
```

Train the modular model:

```bash
python scripts/train_modular.py \
  --data work/datasets/paper_shadow \
  --out work/runs/modular_shadow_exclusion_v1 \
  --epochs 80 \
  --batch-size 64 \
  --base-channels 12
```

Evaluate and sweep thresholds:

```bash
python scripts/evaluate_modular.py \
  --model work/runs/modular_shadow_exclusion_v1/best.keras \
  --data work/datasets/paper_shadow \
  --split val \
  --sweep
```

Export and evaluate the quantized model:

```bash
python scripts/export_tflite.py \
  --model work/runs/modular_shadow_exclusion_v1/best.keras \
  --data work/datasets/paper_shadow \
  --out outputs/paper_shadow_modular_int8.tflite
python scripts/evaluate_tflite_modular.py \
  --model outputs/paper_shadow_modular_int8.tflite \
  --keras-model work/runs/modular_shadow_exclusion_v1/best.keras \
  --data work/datasets/paper_shadow \
  --split val
```

For two-output TFLite models, always check `shadow_output_index` and
`exclude_output_index` from the evaluation report before wiring ESP32
`interpreter->output(i)`. TFLite tensor names can be generic even when Keras
layer names are clear.

## YOLO / Open-Source Teacher Models

YOLOv8-class hand detectors are too large for a normal ESP32-CAM runtime, but
they are useful as offline teacher models on a laptop. Use them to mark hands as
class `2` hard negatives, then train the tiny ESP32 model above:

```bash
pip install ultralytics
python scripts/yolo_hand_teacher.py \
  --input work/datasets/istd_unified \
  --output work/datasets/istd_yolo_handneg \
  --model /path/to/hand-detector.pt \
  --class-names hand,hands \
  --confidence 0.25
```

The script accepts any Ultralytics-compatible YOLO model path or URL. If the
model outputs masks they are used; otherwise detection boxes are converted to
class `2` exclusion masks with dilation.

## Evaluate

```bash
python scripts/evaluate.py \
  --model work/runs/tiny_shadow_v1/best.keras \
  --data work/datasets/paper_shadow \
  --split val
```

## Export Full-Integer TFLite

```bash
python scripts/export_tflite.py \
  --model work/runs/tiny_shadow_v1/best.keras \
  --data work/datasets/paper_shadow \
  --out outputs/paper_shadow_int8.tflite
```

The exported model expects `int8` input shaped `[1, 96, 96, 1]` and emits `int8` logits shaped `[1, 48, 48, 3]`.

## Current Baseline

The strongest checked-in local run is:

```text
work/runs/istd_200_strong_b16
```

It was trained on 200 streamed ISTD samples plus synthetic hard negatives, then exported to:

```text
outputs/paper_shadow_istd200_strong_b16_int8.tflite
```

This is a baseline, not a final model. It proves the full path works and gives a starting point for full-dataset training.

## ESP32-CAM Runtime

See `esp32/README.md` and `esp32/shadowcam_runtime.cpp`.

The intended frame path is:

```text
camera frame -> grayscale 96x96 -> local normalization -> int8 inference
-> shadow probability threshold -> cleanup -> 48x48 mask
```

## Practical Targets

- End-to-end FPS: `10-15 FPS`
- Model size: `40KB-180KB`
- Input: `96x96` grayscale
- Output: `48x48` mask
- Priority: high shadow recall with low hand/object false positives
