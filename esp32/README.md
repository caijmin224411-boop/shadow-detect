# ESP32-CAM Runtime Notes

This folder contains the device-side skeleton for the exported int8 TFLite model.

## Expected Model Contract

```text
input:  int8 [1, 96, 96, 1]
output: int8 [1, 48, 48, 3]
classes: 0 paper/background, 1 shadow, 2 hand/object
```

## Integration Steps

1. Export `outputs/paper_shadow_int8.tflite`.
2. Convert it to a C array:

   ```bash
   python scripts/tflite_to_c_array.py \
     --input outputs/paper_shadow_int8.tflite \
     --output esp32/paper_shadow_model_data.cc
   ```

3. Add `esp32/shadowcam_runtime.cpp` and the generated model data file to an ESP-IDF project with TFLite Micro.
4. Capture OV2640 frames at QQVGA/QCIF-like size, convert to grayscale, downsample to `96x96`.
5. Run local normalization, quantize, invoke TFLite Micro, then call `shadowcam::BuildShadowMask`.

## Performance Knobs

Use these in order if FPS or RAM is too high:

1. Retrain with `--base-channels 6`.
2. Change input to `80x80` and output to `40x40`.
3. Remove skip connections in `src/shadowcam/model.py`.
4. Deploy only every second camera frame and reuse the last mask between frames.

