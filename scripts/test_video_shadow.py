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


def infer_mask(frame_bgr, interpreter, input_detail, shadow_detail, exclude_detail, args):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    norm = local_normalize(small)
    x = norm.astype(np.float32) / 127.5 - 1.0
    x = x[None, ..., None]

    interpreter.set_tensor(input_detail["index"], quantize_input(x, input_detail))
    t0 = time.perf_counter()
    interpreter.invoke()
    infer_ms = (time.perf_counter() - t0) * 1000.0

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
    return mask, infer_ms, float(np.mean(shadow_prob)), float(np.mean(exclude_prob))


def draw_overlay(frame_bgr: np.ndarray, mask48: np.ndarray) -> np.ndarray:
    mask_big = cv2.resize(mask48, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default="outputs/paper_shadow_modular_istd_full_p4_b12_int8.tflite")
    parser.add_argument("--out-dir", default="work/video_shadow_test")
    parser.add_argument("--shadow-threshold", type=float, default=0.70)
    parser.add_argument("--exclude-threshold", type=float, default=0.65)
    parser.add_argument("--shadow-output-index", type=int, default=1)
    parser.add_argument("--exclude-output-index", type=int, default=0)
    parser.add_argument("--min-component-cells", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    shadow_detail = output_details[args.shadow_output_index]
    exclude_detail = output_details[args.exclude_output_index]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"failed_to_open={video_path}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_video = out_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (width, height),
    )

    records = []
    prev_mask = None
    processed = 0
    read_index = 0
    snapshots = {}
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % max(1, args.frame_stride) != 0:
            read_index += 1
            continue

        mask, infer_ms, shadow_mean, exclude_mean = infer_mask(
            frame, interpreter, input_detail, shadow_detail, exclude_detail, args
        )
        if prev_mask is not None:
            persistent = cv2.bitwise_and(prev_mask, cv2.dilate(mask, np.ones((3, 3), np.uint8)))
            mask = cv2.bitwise_or(mask, persistent)
        prev_mask = mask.copy()

        overlay = draw_overlay(frame, mask)
        writer.write(overlay)
        records.append((infer_ms, int(np.count_nonzero(mask)), shadow_mean, exclude_mean))

        if processed in {0, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = path

        processed += 1
        read_index += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)

    summary = (
        f"video={video_path}\n"
        f"source_frames={frame_count}\n"
        f"source_fps={fps:.2f}\n"
        f"size={width}x{height}\n"
        f"processed_frames={processed}\n"
        f"elapsed_s={elapsed:.3f}\n"
        f"pipeline_fps={processed / max(elapsed, 1e-6):.2f}\n"
        f"infer_ms_mean={arr[:, 0].mean() if len(arr) else 0:.2f}\n"
        f"infer_ms_p95={np.percentile(arr[:, 0], 95) if len(arr) else 0:.2f}\n"
        f"mask_cells_last={records[-1][1] if records else 0}\n"
        f"mask_cells_mean={arr[:, 1].mean() if len(arr) else 0:.2f}\n"
        f"mask_cells_max={arr[:, 1].max() if len(arr) else 0:.0f}\n"
        f"shadow_prob_mean_last={records[-1][2] if records else 0:.4f}\n"
        f"exclude_prob_mean_last={records[-1][3] if records else 0:.4f}\n"
        f"overlay_video={out_video.resolve()}\n"
    )
    for idx, path in snapshots.items():
        summary += f"snapshot_{idx}={path.resolve()}\n"
    (out_dir / "summary.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
