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

Open-data second-round training plan added:

- Added `scripts/prepare_shadowtransfer_hf.py` to pull paired `images/masks` directly from the `shadow-transfer-bench/ShadowTransfer` dataset repo file tree. It avoids slow image decoding through the Dataset Viewer and writes the existing unified layout:
  - `train/images`, `train/masks`
  - `val/images`, `val/masks`
  - label `1` is shadow.
- Added `scripts/merge_unified_datasets.py` to combine converted ISTD, SD7K, ShadowTransfer, and negative-augmented datasets without changing labels.
- Added `scripts/synthesize_external_excludes.py` to paste real external hand/object masks from EgoHOS/EgoHands/COCO-like folders onto paper/shadow datasets and mark them as label `2`.
- Added a three-head student model:
  - `shadow_logits`: true shadow
  - `exclude_logits`: hand/object/dark negative
  - `paper_logits`: optional paper gate
- Added `scripts/train_student_three_head.py`, `scripts/evaluate_student_three_head.py`, and `scripts/evaluate_tflite_student_three_head.py`.
- The extended unified mask convention is now:
  - `0`: clean paper or unlabeled non-shadow
  - `1`: shadow
  - `2`: hand/object/dark negative
  - `3`: known non-paper background

Recommended open-data second-round command shape:

```bash
work/.venv/bin/python scripts/prepare_shadowtransfer_hf.py \
  --output work/datasets/shadowtransfer_unified \
  --cache-dir work/hf_cache_shadowtransfer \
  --subset data_loco/fold_2_holdout_chicago \
  --resolution midres \
  --train-limit 1200 \
  --val-limit 300

work/.venv/bin/python scripts/merge_unified_datasets.py \
  --inputs work/datasets/istd_full work/datasets/shadowtransfer_unified work/datasets/sd7k_unified \
  --output work/datasets/open_shadow_v2

work/.venv/bin/python scripts/synthesize_external_excludes.py \
  --input work/datasets/open_shadow_v2 \
  --objects work/datasets/external_object_masks \
  --output work/datasets/open_shadow_v2_excludes \
  --synthetic-objects 3000 \
  --val-synthetic-objects 600

work/.venv/bin/python scripts/train_student_three_head.py \
  --data work/datasets/open_shadow_v2_excludes \
  --out work/runs/student_open_shadow_v2_p4_b12 \
  --epochs 80 \
  --batch-size 64 \
  --base-channels 12 \
  --p4-compatible \
  --shadow-pos-weight 5 \
  --exclude-pos-weight 12
```

SD7K and EgoHOS/EgoHands are still local-manual dataset steps: place converted SD7K at `work/datasets/sd7k_unified` and external masks at `work/datasets/external_object_masks`.

Real-video fine-tuning pass completed:

- Added `scripts/prepare_real_video_review.py` to extract diverse frames from a target mp4 and generate review assets:
  - draft masks from the current three-head Keras model plus brightness/edge heuristics
  - copied editable masks in `masks_reviewed`
  - yellow contour previews
  - `contact_sheet.jpg`
  - `review_manifest.csv`
- Added `scripts/real_video_review_to_unified.py` to convert reviewed masks into unified `train` / `val` / `test` folders.
- Added `scripts/merge_with_real_weight.py` to merge public data with real reviewed data and oversample real training frames.
- Added constrained threshold reporting to `scripts/evaluate_student_three_head.py`; sweep output now includes `constrained_top` using `--min-shadow-recall`, `--max-hand-fpr`, and `--max-dark-fpr`.
- Added `scripts/test_video_shadow_keras.py` for full-video Keras overlay comparison without requiring TFLite export first.
- Source video:
  - `/Users/kongfans/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_zxjeq5ptpdkt22_d12e/temp/RWTemp/2026-06/9849b8961b4b8ab8a44aef4acacc7eba/c9be5e96759298bd43cce0bb156c07e9.mp4`
  - 443 frames, 30 FPS, 1280x720.
- Review package:
  - `work/datasets/real_video_v1_review`
  - extracted frames: `120`
  - contact sheet: `work/datasets/real_video_v1_review/contact_sheet.jpg`
- Auto-reviewed unified dataset:
  - `work/datasets/real_video_v1_unified_auto`
  - train pairs: `84`
  - val pairs: `18`
  - test pairs: `18`
  - mask classes present: `0`, `1`, `2`
  - bad masks: none found
- Weighted merge:
  - `work/datasets/open_shadow_v3_realft_auto`
  - `real_repeat=8`
  - train pairs: `6902`
  - val pairs: `1658`
- Fine-tuned checkpoint:
  - `work/runs/student_open_shadow_v3_realft_auto_p4_b12/best.keras`
  - trained from `work/runs/student_open_shadow_v2_p4_b12/best.keras`
  - early-stopped after 19 logged epochs.
- Real-video auto-test metrics, new fine-tuned model:
  - constrained threshold example: `shadow=0.50`, `exclude=0.70`, `paper=0.10`
  - shadow IoU: `0.283`
  - shadow recall: `0.621`
  - shadow precision: `0.342`
  - hand/object false positive rate: `0.111`
  - dark non-shadow false positive rate: `0.072`
- Real-video auto-test metrics, previous `student_open_shadow_v2_p4_b12` model:
  - constrained threshold example: `shadow=0.40`, `exclude=0.35`, `paper=0.10`
  - shadow IoU: `0.349`
  - shadow recall: `0.640`
  - shadow precision: `0.435`
  - hand/object false positive rate: `0.073`
  - dark non-shadow false positive rate: `0.050`
- Full-video overlay comparison:
  - old model: `work/video_shadow_compare/v2_old/overlay.mp4`
  - new auto-finetuned model: `work/video_shadow_compare/v3_auto/overlay.mp4`
  - old model mean mask cells: `95.90`
  - new model mean mask cells: `132.35`
- Decision:
  - Do not export or deploy `student_open_shadow_v3_realft_auto_p4_b12` to ESP32-P4.
  - It can pass the loose real-video auto-label minimum at a stricter threshold, but it is worse than the previous v2 checkpoint on the same auto-test split and produces larger masks.
  - The most likely cause is noisy automatic pseudo-labels being used as hard ground truth.
  - Next training pass should manually correct the most representative frames in `work/datasets/real_video_v1_review/masks_reviewed`, then rerun `real_video_review_to_unified.py` and train with the corrected masks rather than the draft auto labels.

Second-round open-data seed run completed:

- Verified `scripts/prepare_shadowtransfer_hf.py` with a small ShadowTransfer smoke dataset:
  - `work/datasets/shadowtransfer_smoke`
  - train pairs: `4`
  - val pairs: `2`
- Merged the available full ISTD hard-negative dataset with the ShadowTransfer smoke set:
  - `work/datasets/open_shadow_v2_seed`
  - train pairs: `3834`
  - val pairs: `1042`
- Trained a short three-head P4-compatible seed model:

```bash
work/.venv/bin/python scripts/train_student_three_head.py \
  --data work/datasets/open_shadow_v2_seed \
  --out work/runs/student_open_shadow_v2_seed_p4_b8 \
  --epochs 3 \
  --batch-size 64 \
  --base-channels 8 \
  --p4-compatible \
  --shadow-pos-weight 5 \
  --exclude-pos-weight 12
```

- The 3-epoch seed collapsed toward weak shadow separation: best sweep had usable hand FPR only with low shadow recall.
- Added `--initial-model` support to `scripts/train_student_three_head.py` and continued from the seed with higher shadow weight:

```bash
work/.venv/bin/python scripts/train_student_three_head.py \
  --data work/datasets/open_shadow_v2_seed \
  --out work/runs/student_open_shadow_v2_seed_p4_b8_continue \
  --epochs 5 \
  --batch-size 64 \
  --base-channels 8 \
  --p4-compatible \
  --shadow-pos-weight 10 \
  --exclude-pos-weight 12 \
  --initial-model work/runs/student_open_shadow_v2_seed_p4_b8/best.keras
```

- Final continued Keras validation loss at epoch `5/5`: `val_loss=2.122309`, `val_shadow_logits_loss=1.264360`, `val_exclude_logits_loss=0.905355`, `val_paper_logits_loss=0.101005`.
- Best practical Keras threshold point from the sweep:
  - `shadow_threshold=0.70`, `exclude_threshold=0.40`, `paper_threshold=0.10-0.40`
  - shadow IoU `0.166`, recall `0.215`, precision `0.424`
  - hand FPR `0.078`
  - dark non-shadow FPR `0.052`
- Exported seed int8 TFLite:
  - `outputs/student_open_shadow_v2_seed_p4_b8_int8.tflite`
  - size: `19368` bytes
- TFLite evaluation at `shadow=0.70`, `exclude=0.40`, `paper=0.20`:
  - shadow IoU `0.169`, recall `0.218`, precision `0.433`
  - hand FPR `0.080`
  - dark non-shadow FPR `0.050`
  - desktop TFLite FPS `666.6`
  - output order: `shadow=1`, `exclude=0`, `paper=2`
- Conclusion: the second-round pipeline is now implemented and quantization-safe, but this seed model is not a deployment candidate. Pure ISTD plus a tiny ShadowTransfer smoke sample still under-recognizes shadow. The next actual quality jump needs full ShadowTransfer/SD7K plus real EgoHOS/EgoHands/COCO mask negatives or SAM-generated teacher masks.

Second-round full open-data run completed:

- Converted full available ShadowTransfer paired subsets:
  - `work/datasets/shadowtransfer_loco_midres`: train `450`, val `150`, shadow class non-empty, bad masks `0`
  - `work/datasets/shadowtransfer_chicago_highres`: train `450`, val `150`, shadow class non-empty, bad masks `0`
- Merged open shadow data:
  - inputs: `work/datasets/istd_full`, `work/datasets/shadowtransfer_loco_midres`, `work/datasets/shadowtransfer_chicago_highres`
  - output: `work/datasets/open_shadow_v2`
  - train pairs `2230`, val pairs `840`
- Generated synthetic hard negatives:
  - output: `work/datasets/open_shadow_v2_hardneg`
  - train pairs `6230`, val pairs `1640`
  - class pixels train: `0=1210319012`, `1=245802730`, `2=55019634`
  - class pixels val: `0=343469777`, `1=57121770`, `2=11842885`
  - bad masks `0`
- Local `work/datasets/sd7k_unified` and `work/datasets/external_object_masks` were not present, so SD7K and real hand/object masks were not included in this run.
- Updated `scripts/train_student_three_head.py`:
  - added `--verbose` to allow epoch-level logs
  - appends CSV history when `--initial-model` is used
- Trained main P4-compatible three-head candidate:

```bash
work/.venv/bin/python scripts/train_student_three_head.py \
  --data work/datasets/open_shadow_v2_hardneg \
  --out work/runs/student_open_shadow_v2_p4_b12 \
  --epochs 80 \
  --batch-size 64 \
  --base-channels 12 \
  --p4-compatible \
  --shadow-pos-weight 6 \
  --exclude-pos-weight 10 \
  --paper-loss-weight 0.15
```

- The run was continued from `best.keras` with quiet logging after epoch `7` to avoid per-step log spam:

```bash
work/.venv/bin/python scripts/train_student_three_head.py \
  --data work/datasets/open_shadow_v2_hardneg \
  --out work/runs/student_open_shadow_v2_p4_b12 \
  --epochs 80 \
  --batch-size 64 \
  --base-channels 12 \
  --p4-compatible \
  --shadow-pos-weight 6 \
  --exclude-pos-weight 10 \
  --paper-loss-weight 0.15 \
  --initial-model work/runs/student_open_shadow_v2_p4_b12/best.keras \
  --verbose 2
```

- Best Keras validation checkpoint:
  - `work/runs/student_open_shadow_v2_p4_b12/best.keras`
  - best observed validation loss around `1.6599`
  - desktop Keras evaluation FPS around `420 FPS`
- Threshold sweep best practical points:
  - conservative point: `shadow_threshold=0.60`, `exclude_threshold=0.35`, `paper_threshold=0.10`
    - shadow IoU `0.245`, recall `0.371`, precision `0.418`
    - hand FPR `0.0875`
    - dark non-shadow FPR `0.0759`
  - balanced low-FPR point: `shadow_threshold=0.65`, `exclude_threshold=0.35`, `paper_threshold=0.10`
    - shadow IoU `0.234`, recall `0.332`, precision `0.444`
    - hand FPR `0.0792`
    - dark non-shadow FPR `0.0610`
  - low-threshold recall probe: `shadow_threshold=0.30`, `exclude_threshold=0.35`, `paper_threshold=0.10`
    - shadow IoU `0.230`, recall `0.532`, precision `0.289`
    - hand FPR `0.123`
    - dark non-shadow FPR `0.192`
- Conclusion: this full open-data run is better than the smoke seed, especially on exclude control, but it still fails the deployment minimum:
  - target shadow recall `>=0.55` was not reached together with FPR limits
  - target hand FPR `<=0.12` is only met at thresholds where shadow recall is too low
  - target dark non-shadow FPR `<=0.08` is met only at conservative thresholds where shadow recall is too low
- No `outputs/student_open_shadow_v2_p4_b12_int8.tflite` was exported and this model should not be deployed to ESP32-P4 yet.
- Next data step should prioritize `work/datasets/sd7k_unified` and real external object masks at `work/datasets/external_object_masks`; tuning P4 postprocessing before adding those datasets is unlikely to fix the recall/FPR tradeoff.
