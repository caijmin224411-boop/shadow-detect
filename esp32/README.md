# ESP32-CAM Runtime Notes

This folder contains the device-side skeleton for the exported int8 TFLite model.

## First Real-Device Pass

Start with the Arduino sketch:

```text
esp32/arduino_shadowcam_modular/arduino_shadowcam_modular.ino
```

It targets an AI-Thinker style ESP32-CAM and uses the full-ISTD modular model:

```text
esp32/paper_shadow_modular_istd_full_b12_model_data.cc
```

Default thresholds:

```text
shadow_threshold = 0.70
exclude_threshold = 0.65
```

The sketch prints one serial report per second with FPS, preprocessing time,
inference time, postprocessing time, free heap, free PSRAM, and predicted mask
ratio. Use `FIRST_ROUND_FIELD_TEST.md` to record the first 50-100 scenes.

## Modular Model Contract

```text
input:    int8 [1, 96, 96, 1]
output 0: int8 [1, 48, 48, 1] exclude logits
output 1: int8 [1, 48, 48, 1] shadow logits
```

For `paper_shadow_modular_istd_full_b12_int8.tflite`, the checked output order is:

```text
shadow_output_index = 1
exclude_output_index = 0
```

## Softmax Model Contract

```text
input:  int8 [1, 96, 96, 1]
output: int8 [1, 48, 48, 3]
classes: 0 paper/background, 1 shadow, 2 hand/object
```

## ESP-IDF Integration Steps

1. Export `outputs/paper_shadow_int8.tflite`.
2. Convert it to a C array:

   ```bash
   python scripts/tflite_to_c_array.py \
     --input outputs/paper_shadow_int8.tflite \
     --output esp32/paper_shadow_model_data.cc
   ```

3. Add `esp32/shadowcam_runtime.cpp` and the generated model data file to an ESP-IDF project with TFLite Micro.
4. Capture OV2640 frames at QQVGA/QCIF-like size, convert to grayscale, downsample to `96x96`.
5. Run local normalization, quantize, invoke TFLite Micro, then call `shadowcam::BuildModularShadowMask` for the modular model or `shadowcam::BuildShadowMask` for the softmax model.

The current exported network uses these TFLite ops:

```text
CONV_2D
DEPTHWISE_CONV_2D
CONCATENATION
EXPAND_DIMS
TILE
SHAPE
STRIDED_SLICE
RESHAPE
```

If TFLite Micro reports a missing op while booting, register that op in the
resolver before changing the model.

## Performance Knobs

Use these in order if FPS or RAM is too high:

1. Disable local normalization temporarily and compare FPS / quality.
2. Raise `shadow_threshold` from `0.70` to `0.75` if hand/object false positives dominate.
3. Lower `shadow_threshold` from `0.70` to `0.60` if shadows are mostly missed.
4. Retrain with `--base-channels 8`.
5. Change input to `80x80` and output to `40x40`.
6. Remove skip connections in `src/shadowcam/model.py`.
7. Deploy only every second camera frame and reuse the last mask between frames.
