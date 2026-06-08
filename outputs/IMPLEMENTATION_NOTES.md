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

Full ISTD modular training round completed:

- Created branch `experiment/data-training-v1` for this data/training pass.
- Updated `scripts/prepare_istd_hf.py` with `--target-split` and `--resume`, so interrupted Hugging Face downloads can continue cleanly and the official ISTD `train` / `test` splits can map directly to unified `train` / `val`.
- Downloaded full `Donghyun99/ISTD` into `work/datasets/istd_full`:
  - train pairs: `1330`
  - val pairs: `540`
  - train pixels: class 0 `328039637`, class 1 `80536363`, class 2 `0`
  - val pixels: class 0 `139716473`, class 1 `26171527`, class 2 `0`
- Generated `work/datasets/istd_full_hardneg_v1` with synthetic dark object hard negatives:
  - train pairs: `3830`
  - val pairs: `1040`
  - train pixels: class 0 `912232945`, class 1 `221947863`, class 2 `42395192`
  - val pixels: class 0 `262795030`, class 1 `48170442`, class 2 `8522528`
- Trained the modular two-head model:

```bash
work/.venv/bin/python scripts/train_modular.py --data work/datasets/istd_full_hardneg_v1 --out work/runs/modular_istd_full_b12 --epochs 60 --batch-size 64 --base-channels 12 --shadow-pos-weight 4 --exclude-pos-weight 12
```

- Best checkpoint: `work/runs/modular_istd_full_b12/best.keras`.
- Best validation loss occurred at epoch `52/60`: `val_loss=1.214341`, `val_shadow_logits_loss=0.643151`, `val_exclude_logits_loss=0.708024`.
- Keras threshold points on `istd_full_hardneg_v1` val:
  - high recall point `shadow=0.20`, `exclude=0.75`: shadow IoU `0.439`, recall `0.824`, hand FPR `0.218`, dark-paper FPR `0.156`
  - balanced Keras point `shadow=0.45`, `exclude=0.60`: shadow IoU `0.478`, recall `0.713`, hand FPR `0.139`, dark-paper FPR `0.087`
  - conservative Keras point `shadow=0.55`, `exclude=0.65`: shadow IoU `0.503`, recall `0.709`, hand FPR `0.146`, dark-paper FPR `0.073`
- Exported full-int8 TFLite:
  - `outputs/paper_shadow_modular_istd_full_b12_int8.tflite`
  - size: `54288` bytes
- Generated ESP32 model array:
  - `esp32/paper_shadow_modular_istd_full_b12_model_data.cc`
  - array: `g_paper_shadow_modular_istd_full_b12_model_data`
  - byte length: `54288`
- TFLite output-order verification with `--keras-model`:
  - `shadow_output_index=1`
  - `exclude_output_index=0`
  - output names: `StatefulPartitionedCall_1:1`, `StatefulPartitionedCall_1:0`
  - input quantization: scale `0.007843137718737125`, zero point `-1`
- Recommended int8 default threshold for the first ESP32 integration pass:
  - `shadow_threshold=0.70`
  - `exclude_threshold=0.65`
  - TFLite val metrics: shadow IoU `0.521`, shadow recall `0.673`, shadow precision `0.698`, hand FPR `0.146`, dark-paper FPR `0.052`, desktop TFLite FPS `675`
- Quantization comparison:
  - Keras `0.55/0.65`: shadow IoU `0.503`, recall `0.709`, hand FPR `0.146`
  - TFLite `0.70/0.65`: shadow IoU `0.521`, recall `0.673`, hand FPR `0.146`
  - int8 remains under the `<120KB` target and meets the first-round goal of improving recall beyond `0.444` while keeping hand FPR below `0.15` at the selected threshold.
- Softmax control training completed as a comparison run:

```bash
work/.venv/bin/python scripts/train.py --data work/datasets/istd_full_hardneg_v1 --out work/runs/softmax_istd_full_b16 --epochs 40 --batch-size 64 --base-channels 16 --class-weights 1.0,3.0,12.0
```

- The run was stopped during epoch `29/40` after a usable best checkpoint had already plateaued; `work/runs/softmax_istd_full_b16/best.keras` is from logged epoch `28`.
- Best logged validation values: `val_loss=0.939896`, `val_pixel_acc=0.880079`.
- Keras val metrics for `best.keras`:
  - shadow IoU `0.523`, recall `0.727`, precision `0.651`
  - hand/object IoU `0.317`, hand/object recall `0.826`
  - hand false positive rate `0.143`
  - dark-paper false positive rate `0.069`
- Added `scripts/evaluate_tflite.py` for single-output 3-class full-int8 models.
- Exported full-int8 TFLite:
  - `outputs/paper_shadow_softmax_istd_full_b16_int8.tflite`
  - size: `68488` bytes
- Generated ESP32 model array:
  - `esp32/paper_shadow_softmax_istd_full_b16_model_data.cc`
  - array: `g_paper_shadow_softmax_istd_full_b16_model_data`
  - byte length: `68488`
- TFLite val metrics:
  - shadow IoU `0.526`, recall `0.730`, precision `0.652`
  - hand/object IoU `0.320`, hand/object recall `0.819`
  - hand false positive rate `0.149`
  - dark-paper false positive rate `0.069`
  - desktop TFLite FPS `189`
  - input quantization: scale `0.007843137718737125`, zero point `-1`
  - output quantization: scale `0.07014119625091553`, zero point `32`

First-round model choice:

- `paper_shadow_modular_istd_full_b12_int8.tflite` remains the preferred ESP32 first-pass runtime because the two-head exclusion path gives explicit hand/object suppression thresholds and the model is smaller at `54288` bytes.
- `paper_shadow_softmax_istd_full_b16_int8.tflite` is the strongest offline recall comparison so far, with TFLite shadow recall `0.730` and hand FPR `0.149`.
- Both full-ISTD models beat the previous `istd_200_strong_b16` baseline recall `0.444` while keeping hand FPR under or near the first-round `0.15` target.

First-round ESP32 field-test handoff added:

- Added `shadowcam::LocalNormalize96Integral` to reduce ESP32 preprocessing cost versus the original 9x9 brute-force local mean.
- Added `esp32/arduino_shadowcam_modular/arduino_shadowcam_modular.ino`, an AI-Thinker ESP32-CAM sketch that:
  - captures QQVGA grayscale frames
  - downsamples to `96x96`
  - runs integral-image local normalization
  - invokes `paper_shadow_modular_istd_full_b12`
  - applies `shadow_threshold=0.70` and `exclude_threshold=0.65`
  - prints FPS, preprocessing time, inference time, postprocessing time, mask ratio, heap, and PSRAM once per second
- Added `esp32/FIRST_ROUND_FIELD_TEST.md` for the first 50-100 real scenes.
- Added `scripts/summarize_field_log.py` to summarize serial logs from the field-test sketch.
- Local note: this machine has ESP-IDF but not `arduino-cli`, so the Arduino sketch still needs board-side compile/flash verification in Arduino IDE, Arduino CLI, or PlatformIO.
