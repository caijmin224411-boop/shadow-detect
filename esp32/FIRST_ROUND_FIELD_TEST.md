# First-Round ESP32-CAM Field Test

Use `arduino_shadowcam_modular/arduino_shadowcam_modular.ino` for the first real-device pass.

## Board Setup

- Board: AI Thinker ESP32-CAM or compatible OV2640 board.
- PSRAM: enabled.
- Camera frame: QQVGA grayscale.
- Model: `g_paper_shadow_modular_istd_full_b12_model_data`.
- Default thresholds:
  - `shadow_threshold=0.70`
  - `exclude_threshold=0.65`

## Serial Signals To Record

The sketch prints one line per second:

```text
frames=... fps=... pre_us=... infer_us=... post_us=... mask=... heap=... psram=...
```

Record:

- Average `fps` over 20 seconds after warm-up.
- `infer_us` and `pre_us`, because preprocessing may dominate on ESP32.
- Minimum `heap` and `psram`.
- `mask` ratio for obvious no-shadow, small-shadow, and large-shadow scenes.

If you save serial output to a text file, summarize it with:

```bash
python scripts/summarize_field_log.py work/field_logs/modular_scene_001.txt
```

## First 50-100 Scenes

Capture quick phone photos or screenshots for failures. Keep this split balanced:

- 10 white paper scenes.
- 10 yellow/gray paper scenes.
- 10 printed or handwritten paper scenes.
- 10 hand-on-paper / hand-above-paper scenes.
- 10 pen/ruler/phone dark-object scenes.
- 10 mixed lighting scenes, if time allows.

## Pass Criteria

- End-to-end FPS: `>=10`.
- No crash after 5 minutes.
- Free heap and PSRAM do not keep falling.
- Hand/object dark areas are not consistently marked as shadow.
- Real shadows are detected as coherent regions, even if boundaries are coarse.

## CSV Template

```csv
scene_id,paper_type,lighting,hand_or_object,real_shadow,avg_fps,pre_us,infer_us,post_us,mask_ratio,shadow_quality,hand_false_positive,dark_paper_false_positive,notes
001,white,desk_lamp,none,yes,,,,,,,,,
002,white,desk_lamp,hand_on_paper,no,,,,,,,,,
003,printed,natural,phone_edge,no,,,,,,,,,
```
