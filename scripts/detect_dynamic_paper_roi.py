#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def detect_paper_mask(frame_bgr: np.ndarray, min_area_ratio: float) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    l = lab[..., 0]

    # Paper in this task is usually brighter and less saturated than the table/arm.
    bright = (l > np.percentile(l, 42)).astype(np.uint8)
    low_sat = (s < np.percentile(s, 72)).astype(np.uint8)
    not_too_dark = (v > 55).astype(np.uint8)
    candidate = (bright & low_sat & not_too_dark).astype(np.uint8) * 255

    kernel = np.ones((11, 11), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return np.zeros((height, width), dtype=np.uint8)

    min_area = int(height * width * min_area_ratio)
    scored = []
    cx0, cy0 = width * 0.5, height * 0.55
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx = x + w * 0.5
        cy = y + h_box * 0.5
        center_penalty = ((cx - cx0) / width) ** 2 + ((cy - cy0) / height) ** 2
        score = area * (1.0 - 0.35 * center_penalty)
        scored.append((score, label))
    if not scored:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        label = max(scored, key=lambda item: item[0])[1]

    mask = np.where(labels == label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255)

    # Do not let a bad hull take the whole frame.
    hull_ratio = float(np.count_nonzero(hull_mask)) / max(height * width, 1)
    if hull_ratio > 0.92:
        return mask
    return hull_mask


def smooth_mask(prev: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or prev.shape != current.shape:
        return current
    blended = cv2.addWeighted(prev.astype(np.float32), alpha, current.astype(np.float32), 1.0 - alpha, 0)
    return np.where(blended >= 127, 255, 0).astype(np.uint8)


def draw_preview(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (255, 160, 0)
    overlay = cv2.addWeighted(overlay, 0.80, fill, 0.20, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 200, 0), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out-dir", default="work/video_shadow_compare/dynamic_paper_roi")
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed_to_open={args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_dir / "paper_roi_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev = None
    records = []
    snapshots = {}
    processed = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = detect_paper_mask(frame, args.min_area_ratio)
        mask = smooth_mask(prev, mask, args.smooth_alpha)
        prev = mask
        preview = draw_preview(frame, mask)
        writer.write(preview)
        cv2.imwrite(str(masks_dir / f"frame_{processed:05d}.png"), mask)
        if processed in {0, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_paper_roi.jpg"
            cv2.imwrite(str(path), preview)
            snapshots[processed] = str(path.resolve())
        records.append(int(np.count_nonzero(mask)))
        processed += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)
    summary = {
        "video": str(Path(args.video).resolve()),
        "source_frames": source_frames,
        "processed_frames": processed,
        "pipeline_fps": processed / max(elapsed, 1e-6),
        "mask_pixels_mean": float(arr.mean()) if len(arr) else 0.0,
        "mask_pixels_min": int(arr.min()) if len(arr) else 0,
        "mask_pixels_max": int(arr.max()) if len(arr) else 0,
        "overlay_video": str((out_dir / "paper_roi_overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
