# Implementation Notes

This workspace now contains a complete first-pass implementation for the ESP32-CAM paper-shadow segmentation plan.

Start with:

```bash
python3 -m venv work/.venv
source work/.venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_istd_hf.py --dataset Donghyun99/ISTD --output work/datasets/istd_unified
python scripts/synthesize_hard_negatives.py --input work/datasets/istd_unified --output work/datasets/paper_shadow --synthetic-objects 1200
python scripts/train.py --data work/datasets/paper_shadow --out work/runs/tiny_shadow_v1 --epochs 80 --batch-size 64
python scripts/export_tflite.py --model work/runs/tiny_shadow_v1/best.keras --data work/datasets/paper_shadow --out outputs/paper_shadow_int8.tflite
```

If the Hugging Face dataset download is slow, first run the synthetic smoke test:

```bash
python scripts/generate_synthetic_paper_dataset.py --output work/datasets/synthetic_paper_shadow --train 160 --val 40
python scripts/train.py --data work/datasets/synthetic_paper_shadow --out work/runs/smoke --epochs 2 --batch-size 16
```

The ESP32 integration notes are in `esp32/README.md`.

Smoke verification completed in this workspace:

- Built Tiny Depthwise U-Net: input `[96,96,1]`, output `[48,48,3]`, 8,107 trainable parameters.
- Generated a synthetic paper-shadow dataset with 32 train and 8 validation samples.
- Ran one smoke-training epoch and saved `work/runs/smoke/best.keras`.
- Exported `outputs/paper_shadow_smoke_int8.tflite`, 38,992 bytes.
- Generated `esp32/paper_shadow_model_data.cc` from the smoke TFLite model.
- Verified Python syntax and ESP32 C++ syntax.

The smoke model is only a pipeline check, not a real-quality shadow detector.

Hugging Face network note:

- `scripts/inspect_hf_dataset.py` confirmed `Donghyun99/ISTD` is reachable and has `train` / `test` splits.
- `scripts/prepare_istd_hf.py` now uses streaming by default so small validation downloads do not prepare the full parquet dataset.
- The local HF client may still warn about LibreSSL on macOS; the current workflow succeeds with cache isolation under `work/hf_cache3`.

ISTD real-data smoke verification completed:

- Streamed 20 samples from `Donghyun99/ISTD` into `work/datasets/istd_unified_sample`.
- Built `work/datasets/istd_hardneg_sample` with 20 synthetic hard-negative object overlays.
- Ran one training epoch and saved `work/runs/istd_smoke/best.keras`.
- Exported `outputs/paper_shadow_istd_smoke_int8.tflite`, 38,992 bytes.
- Generated `esp32/paper_shadow_istd_smoke_model_data.cc`.
- This is still a smoke model: one epoch on 20 real samples is not enough to judge recognition quality.

ISTD-200 baseline completed:

- Streamed 200 samples from `Donghyun99/ISTD` into `work/datasets/istd_unified_200`.
- Built `work/datasets/istd_hardneg_200_strong` with 500 train and 120 validation synthetic hard-negative overlays.
- Trained `work/runs/istd_200_strong_b16` with `base_channels=16`, 26,579 trainable parameters, and class weights `1.0,3.0,12.0`.
- Best Keras validation metrics on `istd_hardneg_200_strong`:
  - paper/background IoU: `0.761`
  - shadow IoU: `0.316`
  - shadow recall: `0.444`
  - shadow precision: `0.524`
  - hand/object IoU: `0.317`
  - hand/object recall: `0.761`
  - hand false positive rate into shadow: `0.169`
- Shadow probability threshold scan:
  - threshold `0.30`: shadow IoU `0.327`, recall `0.533`, hand false positive `0.256`
  - threshold `0.60`: shadow IoU `0.271`, recall `0.324`, hand false positive `0.095`
  - threshold `0.70`: shadow IoU `0.229`, recall `0.255`, hand false positive `0.057`
- Exported `outputs/paper_shadow_istd200_strong_b16_int8.tflite`, 68,488 bytes.
- Generated `esp32/paper_shadow_istd200_strong_b16_model_data.cc`.
- This is the current best balanced baseline in the workspace, but it is still below the final target of `>=85%` shadow recall and `<5%` hand false positive.

Modular hand/object exclusion experiment added:

- Added `build_modular_shadow_exclusion_net`, a two-head tiny network with shared features and separate `shadow_logits` / `exclude_logits` outputs.
- Added `scripts/train_modular.py` for weighted binary shadow + exclusion training.
- Added `scripts/evaluate_modular.py` to sweep shadow/exclusion thresholds and report hand false positives directly.
- Added `scripts/evaluate_tflite_modular.py` to verify full-int8 two-output TFLite models instead of trusting only Keras metrics.
- Added `BuildModularShadowMask` in `esp32/shadowcam_runtime.cpp`; it converts probability thresholds to int8 logit thresholds once and then applies a cheap per-pixel compare.
- Added optional `scripts/yolo_hand_teacher.py`; YOLOv8-style hand detectors should be used on the computer as hard-negative/teacher labelers, not as the ESP32 runtime model.
- TFLite may expose generic output names and a surprising output order; run `scripts/evaluate_tflite_modular.py` with `--keras-model` and use the reported `shadow_output_index` / `exclude_output_index`.

Recommended next modular run:

```bash
python scripts/train_modular.py --data work/datasets/istd_hardneg_200_strong --out work/runs/modular_istd200_b12 --epochs 40 --batch-size 64 --base-channels 12 --shadow-pos-weight 4 --exclude-pos-weight 12
python scripts/evaluate_modular.py --model work/runs/modular_istd200_b12/best.keras --data work/datasets/istd_hardneg_200_strong --split val --sweep
python scripts/export_tflite.py --model work/runs/modular_istd200_b12/best.keras --data work/datasets/istd_hardneg_200_strong --out outputs/paper_shadow_modular_istd200_b12_int8.tflite
python scripts/evaluate_tflite_modular.py --model outputs/paper_shadow_modular_istd200_b12_int8.tflite --keras-model work/runs/modular_istd200_b12/best.keras --data work/datasets/istd_hardneg_200_strong --split val
```

Modular smoke verification completed:

- Trained a 1-epoch fixed-order smoke model at `work/runs/modular_smoke_order_b8`.
- Exported `outputs/paper_shadow_modular_order_smoke_b8_int8.tflite`, 40,840 bytes.
- TFLite/Keras output-order alignment reported `shadow_output_index=1` and `exclude_output_index=0` for that smoke export.
- The smoke model is intentionally undertrained and not a quality detector; it is only a two-head pipeline check.
