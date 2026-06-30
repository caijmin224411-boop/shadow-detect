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


def student_probs(model, frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 127.5 - 1.0
    shadow_logits, exclude_logits, paper_logits = unpack_outputs(model(x[None, ..., None], training=False))
    shadow = tf.nn.sigmoid(shadow_logits).numpy()[0, ..., 0]
    exclude = tf.nn.sigmoid(exclude_logits).numpy()[0, ..., 0]
    paper = tf.nn.sigmoid(paper_logits).numpy()[0, ..., 0]
    size = (frame_bgr.shape[1], frame_bgr.shape[0])
    return (
        cv2.resize(shadow, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(exclude, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(paper, size, interpolation=cv2.INTER_LINEAR),
    )


def clean_mask(mask: np.ndarray, min_area: int):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(overlay, 0.82, fill, 0.18, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--bdrar-mask-dir", required=True)
    parser.add_argument("--student-model", default="work/runs/student_open_shadow_v2_p4_b12/best.keras")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--paper-threshold", type=float, default=0.20)
    parser.add_argument("--exclude-threshold", type=float, default=0.45)
    parser.add_argument("--student-shadow-threshold", type=float, default=0.20)
    parser.add_argument("--require-student-shadow", action="store_true")
    parser.add_argument("--roi-mask", default="", help="Optional full-resolution paper ROI mask. Nonzero pixels are allowed.")
    parser.add_argument("--roi-mask-dir", default="", help="Optional per-frame paper ROI masks named frame_00000.png.")
    parser.add_argument("--min-area", type=int, default=900)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.student_model, compile=False)
    roi_mask = None
    if args.roi_mask:
        roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            raise FileNotFoundError(args.roi_mask)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed_to_open={args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    records = []
    snapshots = {}
    processed = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bdrar_path = Path(args.bdrar_mask_dir) / f"frame_{processed:05d}.png"
        bdrar_mask = cv2.imread(str(bdrar_path), cv2.IMREAD_GRAYSCALE)
        if bdrar_mask is None:
            break
        frame_roi = roi_mask
        if args.roi_mask_dir:
            roi_path = Path(args.roi_mask_dir) / f"frame_{processed:05d}.png"
            frame_roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
            if frame_roi is None:
                break
        if frame_roi is not None and frame_roi.shape[:2] != frame.shape[:2]:
            frame_roi = cv2.resize(frame_roi, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        shadow_prob, exclude_prob, paper_prob = student_probs(model, frame)
        gate = (paper_prob >= args.paper_threshold) & (exclude_prob < args.exclude_threshold)
        if args.require_student_shadow:
            gate &= shadow_prob >= args.student_shadow_threshold
        if frame_roi is not None:
            gate &= frame_roi > 0
        mask = np.where((bdrar_mask > 0) & gate, 255, 0).astype(np.uint8)
        mask = clean_mask(mask, args.min_area)
        overlay = draw_overlay(frame, mask)
        writer.write(overlay)
        cv2.imwrite(str(masks_dir / f"frame_{processed:05d}.png"), mask)
        if processed in {0, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = str(path.resolve())
        records.append((int(np.count_nonzero(mask)), float(paper_prob.mean()), float(exclude_prob.mean())))
        processed += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)
    summary = {
        "video": str(Path(args.video).resolve()),
        "bdrar_mask_dir": str(Path(args.bdrar_mask_dir).resolve()),
        "student_model": str(Path(args.student_model).resolve()),
        "paper_threshold": args.paper_threshold,
        "exclude_threshold": args.exclude_threshold,
        "require_student_shadow": args.require_student_shadow,
        "roi_mask": str(Path(args.roi_mask).resolve()) if args.roi_mask else "",
        "roi_mask_dir": str(Path(args.roi_mask_dir).resolve()) if args.roi_mask_dir else "",
        "source_frames": source_frames,
        "processed_frames": processed,
        "elapsed_s": elapsed,
        "pipeline_fps": processed / max(elapsed, 1e-6),
        "mask_pixels_mean": float(arr[:, 0].mean()) if len(arr) else 0.0,
        "mask_pixels_max": int(arr[:, 0].max()) if len(arr) else 0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
