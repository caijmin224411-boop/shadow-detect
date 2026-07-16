#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["shadow_logits"], outputs["exclude_logits"], outputs["paper_logits"]
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        by_name = {getattr(t, "name", "").split("/")[0]: t for t in outputs}
        if {"shadow_logits", "exclude_logits", "paper_logits"} <= set(by_name):
            return by_name["shadow_logits"], by_name["exclude_logits"], by_name["paper_logits"]
        return outputs[0], outputs[1], outputs[2]
    raise TypeError(f"Expected three model outputs, got {type(outputs)!r}")


def local_normalize(gray96: np.ndarray) -> np.ndarray:
    src = gray96.astype(np.float32)
    mean = cv2.blur(src, (9, 9), borderType=cv2.BORDER_REPLICATE)
    return np.clip(src - mean + 128.0, 0, 255).astype(np.uint8)


def remove_yellow_overlay(frame: np.ndarray) -> np.ndarray:
    """Inpaint old yellow comparison contours in the retained proxy video."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray((20, 145, 145)), np.asarray((42, 255, 255)))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if not mask.any():
        return frame
    return cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA)


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


def infer_mask(frame_bgr: np.ndarray, model, args):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    if args.local_normalize:
        small = local_normalize(small)
    x = small.astype(np.float32) / 127.5 - 1.0
    x = x[None, ..., None]

    t0 = time.perf_counter()
    outputs = model(x, training=False)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    if not isinstance(outputs, (dict, list, tuple)) and outputs.shape[-1] == 3:
        probs = tf.nn.softmax(outputs, axis=-1).numpy()[0]
        shadow_prob = probs[..., 1]
        exclude_prob = probs[..., 2]
        paper_prob = probs[..., 0] + probs[..., 1]
        mask = np.argmax(probs, axis=-1) == 1
    else:
        shadow_logits, exclude_logits, paper_logits = unpack_outputs(outputs)
        shadow_prob = tf.nn.sigmoid(shadow_logits).numpy()[0, ..., 0]
        exclude_prob = tf.nn.sigmoid(exclude_logits).numpy()[0, ..., 0]
        paper_prob = tf.nn.sigmoid(paper_logits).numpy()[0, ..., 0]
        mask = (
            (shadow_prob >= args.shadow_threshold)
            & (exclude_prob < args.exclude_threshold)
            & (paper_prob >= args.paper_threshold)
        )
    mask = postprocess_mask(mask, args.min_component_cells)
    return mask, infer_ms, float(np.mean(shadow_prob)), float(np.mean(exclude_prob)), float(np.mean(paper_prob))


def draw_overlay(frame_bgr: np.ndarray, mask48: np.ndarray, fill_alpha: float) -> np.ndarray:
    mask_big = cv2.resize(mask48, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    if fill_alpha > 0:
        tint = frame_bgr.copy()
        tint[mask_big > 0] = (0, 255, 255)
        overlay = cv2.addWeighted(tint, fill_alpha, overlay, 1.0 - fill_alpha, 0.0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--exclude-threshold", type=float, default=0.55)
    parser.add_argument("--paper-threshold", type=float, default=0.10)
    parser.add_argument("--min-component-cells", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--local-normalize", action="store_true")
    parser.add_argument("--remove-yellow-overlay", action="store_true")
    parser.add_argument("--fill-alpha", type=float, default=0.22)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model, compile=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"failed_to_open={video_path}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (width, height),
    )

    records = []
    processed = 0
    read_index = 0
    snapshot_indices = {0, 30, 90, 150, 240, 360}
    snapshots = {}
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.remove_yellow_overlay:
            frame = remove_yellow_overlay(frame)
        if read_index % max(1, args.frame_stride) != 0:
            read_index += 1
            continue
        mask, infer_ms, shadow_mean, exclude_mean, paper_mean = infer_mask(frame, model, args)
        overlay = draw_overlay(frame, mask, args.fill_alpha)
        writer.write(overlay)
        mask_cells = int(np.count_nonzero(mask))
        records.append((infer_ms, mask_cells, shadow_mean, exclude_mean, paper_mean))
        if processed in snapshot_indices:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = str(path.resolve())
        processed += 1
        read_index += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)
    summary = {
        "video": str(video_path),
        "model": str(Path(args.model).resolve()),
        "source_frames": source_frames,
        "source_fps": fps,
        "size": [width, height],
        "processed_frames": processed,
        "elapsed_s": elapsed,
        "pipeline_fps": processed / max(elapsed, 1e-6),
        "infer_ms_mean": float(arr[:, 0].mean()) if len(arr) else 0.0,
        "infer_ms_p95": float(np.percentile(arr[:, 0], 95)) if len(arr) else 0.0,
        "mask_cells_mean": float(arr[:, 1].mean()) if len(arr) else 0.0,
        "mask_cells_max": int(arr[:, 1].max()) if len(arr) else 0,
        "shadow_prob_mean_last": float(records[-1][2]) if records else 0.0,
        "exclude_prob_mean_last": float(records[-1][3]) if records else 0.0,
        "paper_prob_mean_last": float(records[-1][4]) if records else 0.0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
