#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def quantize_input(x: np.ndarray, input_detail) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(input_detail["dtype"])


def dequantize_output(y: np.ndarray, output_detail) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    return (y.astype(np.float32) - zero_point) * scale


def local_normalize(gray96: np.ndarray) -> np.ndarray:
    src = gray96.astype(np.float32)
    mean = cv2.blur(src, (9, 9), borderType=cv2.BORDER_REPLICATE)
    return np.clip(src - mean + 128.0, 0, 255).astype(np.uint8)


def postprocess_mask(mask48: np.ndarray, min_component_cells: int) -> np.ndarray:
    mask = mask48.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_cells:
            cleaned[labels == label] = 255
    return cleaned


def open_camera(max_index: int):
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return index, cap
        cap.release()
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/paper_shadow_modular_istd_full_p4_b12_int8.tflite")
    parser.add_argument("--out", default="work/camera_shadow_test")
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--shadow-threshold", type=float, default=0.70)
    parser.add_argument("--exclude-threshold", type=float, default=0.65)
    parser.add_argument("--shadow-output-index", type=int, default=1)
    parser.add_argument("--exclude-output-index", type=int, default=0)
    parser.add_argument("--min-component-cells", type=int, default=5)
    parser.add_argument("--max-camera-index", type=int, default=4)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    shadow_detail = output_details[args.shadow_output_index]
    exclude_detail = output_details[args.exclude_output_index]

    camera_index, cap = open_camera(args.max_camera_index)
    if cap is None:
        print("NO_CAMERA_OPENED")
        print("On macOS, grant camera access to the app running this script, then rerun.")
        return 2

    for _ in range(args.warmup):
        cap.read()
        time.sleep(0.03)

    records = []
    raw = None
    overlay = None
    prev_mask = None
    start = time.perf_counter()

    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        raw = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        norm = local_normalize(small)
        x = norm.astype(np.float32) / 127.5 - 1.0
        x = x[None, ..., None]

        interpreter.set_tensor(input_detail["index"], quantize_input(x, input_detail))
        infer_start = time.perf_counter()
        interpreter.invoke()
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        shadow_logits = dequantize_output(
            interpreter.get_tensor(shadow_detail["index"]), shadow_detail
        )[0, ..., 0]
        exclude_logits = dequantize_output(
            interpreter.get_tensor(exclude_detail["index"]), exclude_detail
        )[0, ..., 0]
        shadow_prob = 1.0 / (1.0 + np.exp(-shadow_logits))
        exclude_prob = 1.0 / (1.0 + np.exp(-exclude_logits))
        mask = (shadow_prob >= args.shadow_threshold) & (exclude_prob < args.exclude_threshold)
        mask = postprocess_mask(mask, args.min_component_cells)

        if prev_mask is not None:
            persistent = cv2.bitwise_and(prev_mask, cv2.dilate(mask, np.ones((3, 3), np.uint8)))
            mask = cv2.bitwise_or(mask, persistent)
        prev_mask = mask.copy()

        mask_big = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        records.append(
            (
                infer_ms,
                int(np.count_nonzero(mask)),
                float(np.mean(shadow_prob)),
                float(np.mean(exclude_prob)),
            )
        )

    elapsed = time.perf_counter() - start
    cap.release()

    if raw is None or overlay is None or prev_mask is None:
        print("NO_FRAMES_CAPTURED")
        return 3

    raw_path = out_dir / "latest_raw.jpg"
    overlay_path = out_dir / "latest_overlay.jpg"
    mask_path = out_dir / "latest_mask48.png"
    summary_path = out_dir / "summary.txt"
    cv2.imwrite(str(raw_path), raw)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), prev_mask)

    arr = np.asarray(records, dtype=np.float32)
    summary = (
        f"camera_index={camera_index}\n"
        f"frames={len(records)}\n"
        f"elapsed_s={elapsed:.3f}\n"
        f"pipeline_fps={len(records) / max(elapsed, 1e-6):.2f}\n"
        f"infer_ms_mean={arr[:, 0].mean() if len(arr) else 0:.2f}\n"
        f"infer_ms_p95={np.percentile(arr[:, 0], 95) if len(arr) else 0:.2f}\n"
        f"mask_cells_last={records[-1][1] if records else 0}\n"
        f"mask_cells_mean={arr[:, 1].mean() if len(arr) else 0:.2f}\n"
        f"shadow_prob_mean_last={records[-1][2] if records else 0:.4f}\n"
        f"exclude_prob_mean_last={records[-1][3] if records else 0:.4f}\n"
        f"raw={raw_path.resolve()}\n"
        f"overlay={overlay_path.resolve()}\n"
        f"mask={mask_path.resolve()}\n"
    )
    summary_path.write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
