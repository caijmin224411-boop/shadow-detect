#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    if shape is not None and image.shape[:2] != shape:
        h, w = shape
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return image


def paper_inner_roi(roi: np.ndarray, erode: int) -> np.ndarray:
    binary = (roi > 0).astype(np.uint8) * 255
    if erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def remove_bad_components(mask: np.ndarray, roi: np.ndarray, min_area: int, max_area_ratio: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    h, w = mask.shape[:2]
    max_area = int(h * w * max_area_ratio)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = labels == label
        roi_overlap = np.count_nonzero(comp & (roi > 0)) / max(1, area)
        if roi_overlap < 0.85:
            continue
        cleaned[comp] = 255
    return cleaned


def skin_like_exclude(frame_bgr: np.ndarray, roi: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycrcb)
    h, s, v = cv2.split(hsv)
    skin = (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 135)
        & (s >= 20)
        & (v >= 50)
        & (roi > 0)
    )
    out = skin.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    out = cv2.dilate(out, kernel, iterations=1)
    return out


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(frame_bgr, 0.86, fill, 0.14, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--roi-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-area", type=int, default=1800)
    parser.add_argument("--max-area-ratio", type=float, default=0.30)
    parser.add_argument("--roi-erode", type=int, default=3)
    parser.add_argument("--skin-exclude", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mask_dir = Path(args.mask_dir)
    roi_dir = Path(args.roi_mask_dir)
    snapshots = {}
    records = []
    start = time.perf_counter()
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask_path = mask_dir / f"frame_{index:05d}.png"
        roi_path = roi_dir / f"frame_{index:05d}.png"
        if not mask_path.exists() or not roi_path.exists():
            break
        mask = read_gray(mask_path, (height, width))
        roi = paper_inner_roi(read_gray(roi_path, (height, width)), args.roi_erode)
        filtered = ((mask > 0) & (roi > 0)).astype(np.uint8) * 255
        skin = skin_like_exclude(frame, roi, args.skin_exclude)
        filtered[skin > 0] = 0
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        filtered = remove_bad_components(filtered, roi, args.min_area, args.max_area_ratio)
        cv2.imwrite(str(masks_dir / f"frame_{index:05d}.png"), filtered)
        overlay = draw_overlay(frame, filtered)
        writer.write(overlay)
        if index in {0, 1, 10, 30, 50, 90, 120, 150, 240, 300, 360, 420}:
            path = out_dir / f"frame_{index:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[index] = str(path.resolve())
        records.append(int(np.count_nonzero(filtered)))
        index += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    summary = {
        "video": str(Path(args.video).resolve()),
        "input_mask_dir": str(mask_dir.resolve()),
        "roi_mask_dir": str(roi_dir.resolve()),
        "source_frames": source_frames,
        "processed_frames": index,
        "fps": fps,
        "size": [width, height],
        "elapsed_s": elapsed,
        "pipeline_fps": index / max(elapsed, 1e-6),
        "mask_pixels_mean": float(np.mean(records)) if records else 0.0,
        "mask_pixels_max": int(np.max(records)) if records else 0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
