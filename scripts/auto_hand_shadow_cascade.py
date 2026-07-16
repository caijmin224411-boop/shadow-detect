#!/usr/bin/env python3
"""Automatic hand-guided shadow filtering without manual masks.

The hand detector is used as a soft spatial prior. If the hand is missed, the
pipeline falls back to the paper-constrained shadow proposal instead of forcing
an empty result.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


FRAME_RE = re.compile(r"frame_(\d+)")


def frame_index(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    if not match:
        raise ValueError(f"cannot read frame index from {path.name}")
    return int(match.group(1))


def read_indexed_mask(mask_dir: Path | None, index: int, shape: tuple[int, int]) -> np.ndarray:
    if mask_dir is None:
        return np.full(shape, 255, dtype=np.uint8)
    candidates = [
        mask_dir / f"frame_{index:05d}.png",
        mask_dir / f"frame_{index:04d}.png",
        mask_dir / f"frame_{index:05d}.jpg",
        mask_dir / f"frame_{index:04d}.jpg",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return np.full(shape, 255, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.full(shape, 255, dtype=np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask >= 127, 255, 0).astype(np.uint8)


def skin_fallback(frame: np.ndarray, min_area: int) -> np.ndarray:
    """Conservative fallback for frames where landmark detection misses a hand."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, cr, cb = cv2.split(ycrcb)
    _, sat, value = cv2.split(hsv)
    mask = (
        (cr >= 132)
        & (cr <= 182)
        & (cb >= 72)
        & (cb <= 137)
        & (sat >= 24)
        & (value >= 45)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a binary mask in linear time, including for large radii."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    background = np.where(mask > 0, 0, 1).astype(np.uint8)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_3)
    return np.where(distance <= float(radius), 255, 0).astype(np.uint8)


def detect_hand_mask(frame: np.ndarray, hands, hand_dilate: int, allow_skin_fallback: bool) -> tuple[np.ndarray, int, str]:
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    mask = np.zeros((height, width), dtype=np.uint8)
    count = 0
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            points = np.asarray(
                [
                    (int(np.clip(item.x, 0, 1) * (width - 1)), int(np.clip(item.y, 0, 1) * (height - 1)))
                    for item in landmarks.landmark
                ],
                dtype=np.int32,
            )
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
            count += 1
    source = "mediapipe" if mask.any() else "none"
    if mask.any():
        # Landmarks cover the palm reliably but can leave skin around curled
        # fingers. Expand only skin components close to a confirmed hand.
        skin = skin_fallback(frame, min_area=max(450, height * width // 1800))
        proximity = expand_mask(mask, 60)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(skin, proximity))
    if not mask.any() and allow_skin_fallback:
        mask = skin_fallback(frame, min_area=max(900, height * width // 900))
        if mask.any():
            count = 1
            source = "skin_fallback"
        else:
            source = "none"
    if mask.any() and hand_dilate > 0:
        mask = expand_mask(mask, hand_dilate)
    return mask, count, source


def clean_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 255
    return output


def remove_yellow_overlay(frame: np.ndarray) -> np.ndarray:
    """Inpaint thin yellow contours from a retained proxy video."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray((20, 145, 145)), np.asarray((42, 255, 255)))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if not mask.any():
        return frame
    return cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA)


def filter_shadow(
    candidate: np.ndarray,
    paper: np.ndarray,
    hand: np.ndarray,
    prev_shadow: np.ndarray | None,
    search_radius: int,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    base = cv2.bitwise_and(candidate, paper)
    if hand.any():
        radius = max(1, search_radius)
        near_hand = expand_mask(hand, radius)
        allowed = near_hand
        mode = "hand_guided"
        if prev_shadow is not None and prev_shadow.any():
            temporal = expand_mask(prev_shadow, 15)
            allowed = cv2.bitwise_or(allowed, temporal)
        filtered = cv2.bitwise_and(base, allowed)
        filtered[hand > 0] = 0
    else:
        near_hand = np.zeros_like(base)
        filtered = base
        mode = "soft_fallback"
        if prev_shadow is not None and prev_shadow.any():
            temporal = expand_mask(prev_shadow, 25)
            temporal_keep = cv2.bitwise_and(base, temporal)
            filtered = cv2.bitwise_or(temporal_keep, cv2.bitwise_and(base, paper))
    return clean_components(filtered, min_area), near_hand, mode


def tint(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.28) -> np.ndarray:
    output = frame.copy()
    layer = np.zeros_like(frame)
    layer[mask > 0] = color
    output = cv2.addWeighted(output, 1.0, layer, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(output, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def comparison_panel(
    frame: np.ndarray,
    candidate: np.ndarray,
    hand: np.ndarray,
    near: np.ndarray,
    filtered: np.ndarray,
    mode: str,
) -> np.ndarray:
    original = label(frame, "1 Input frame")
    hand_view = tint(frame, near, (0, 150, 0), 0.12)
    hand_view = tint(hand_view, hand, (0, 0, 255), 0.30)
    hand_view = label(hand_view, f"2 Auto hand + search ROI ({mode})")
    baseline = label(tint(frame, candidate, (0, 215, 255)), "3 Baseline shadow proposal")
    result = label(tint(frame, filtered, (255, 255, 0)), "4 Hand-guided shadow result")
    height, width = frame.shape[:2]
    cell_size = (width // 2, height // 2)
    cells = [cv2.resize(item, cell_size, interpolation=cv2.INTER_AREA) for item in (original, hand_view, baseline, result)]
    return np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))


def process_frame(
    frame: np.ndarray,
    index: int,
    hands,
    candidate_dir: Path,
    roi_dir: Path | None,
    prev_shadow: np.ndarray | None,
    args,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if args.remove_yellow_overlay:
        frame = remove_yellow_overlay(frame)
    shape = frame.shape[:2]
    candidate = read_indexed_mask(candidate_dir, index, shape)
    paper = read_indexed_mask(roi_dir, index, shape)
    hand, hand_count, hand_source = detect_hand_mask(frame, hands, args.hand_dilate, args.skin_fallback)
    filtered, near, mode = filter_shadow(
        candidate,
        paper,
        hand,
        prev_shadow,
        args.search_radius,
        args.min_area,
    )
    candidate_paper = cv2.bitwise_and(candidate, paper)
    panel = comparison_panel(frame, candidate_paper, hand, near, filtered, mode)
    base_pixels = int(np.count_nonzero(candidate_paper))
    result_pixels = int(np.count_nonzero(filtered))
    candidate_hand_overlap = int(np.count_nonzero(cv2.bitwise_and(candidate_paper, hand)))
    result_hand_overlap = int(np.count_nonzero(cv2.bitwise_and(filtered, hand)))
    record = {
        "index": index,
        "hand_count": hand_count,
        "hand_source": hand_source,
        "mode": mode,
        "candidate_pixels": base_pixels,
        "result_pixels": result_pixels,
        "candidate_hand_overlap_pixels": candidate_hand_overlap,
        "result_hand_overlap_pixels": result_hand_overlap,
        "retention_ratio": result_pixels / max(base_pixels, 1),
    }
    return panel, filtered, record


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 2) -> None:
    images = [cv2.imread(str(path)) for path in paths]
    images = [item for item in images if item is not None]
    if not images:
        return
    width = 960
    resized = [cv2.resize(item, (width, int(item.shape[0] * width / item.shape[1])), interpolation=cv2.INTER_AREA) for item in images]
    cell_h = max(item.shape[0] for item in resized)
    rows = []
    for start in range(0, len(resized), columns):
        row_items = resized[start : start + columns]
        while len(row_items) < columns:
            row_items.append(np.full((cell_h, width, 3), 245, dtype=np.uint8))
        row_items = [cv2.copyMakeBorder(item, 0, cell_h - item.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for item in row_items]
        rows.append(np.hstack(row_items))
    cv2.imwrite(str(output), np.vstack(rows))


def run_images(args, hands, out_dir: Path) -> list[dict]:
    paths = sorted(Path(args.images_dir).glob(args.image_glob), key=frame_index)
    records = []
    outputs = []
    prev_shadow = None
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        index = frame_index(path)
        panel, prev_shadow, record = process_frame(
            frame,
            index,
            hands,
            Path(args.candidate_mask_dir),
            Path(args.roi_mask_dir) if args.roi_mask_dir else None,
            prev_shadow,
            args,
        )
        output = out_dir / f"frame_{index:04d}_comparison.jpg"
        cv2.imwrite(str(output), panel)
        outputs.append(output)
        records.append(record)
    make_contact_sheet(outputs, out_dir / "contact_sheet.jpg")
    return records


def run_video(args, hands, out_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    records = []
    snapshots = []
    prev_shadow = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panel, prev_shadow, record = process_frame(
            frame,
            index,
            hands,
            Path(args.candidate_mask_dir),
            Path(args.roi_mask_dir) if args.roi_mask_dir else None,
            prev_shadow,
            args,
        )
        writer.write(panel)
        if index in {0, 30, 90, 150, 240, 360, 420}:
            path = out_dir / f"frame_{index:04d}_comparison.jpg"
            cv2.imwrite(str(path), panel)
            snapshots.append(path)
        records.append(record)
        index += 1
        if args.max_frames and index >= args.max_frames:
            break
    cap.release()
    writer.release()
    make_contact_sheet(snapshots, out_dir / "contact_sheet.jpg")
    return records


def summarize(records: list[dict], elapsed: float, args) -> dict:
    detected = [item for item in records if item["hand_source"] != "none"]
    media = [item for item in records if item["hand_source"] == "mediapipe"]
    base = sum(item["candidate_pixels"] for item in records)
    result = sum(item["result_pixels"] for item in records)
    candidate_hand_overlap = sum(item["candidate_hand_overlap_pixels"] for item in records)
    result_hand_overlap = sum(item["result_hand_overlap_pixels"] for item in records)
    return {
        "input": args.video or args.images_dir,
        "candidate_mask_dir": str(Path(args.candidate_mask_dir).resolve()),
        "roi_mask_dir": str(Path(args.roi_mask_dir).resolve()) if args.roi_mask_dir else "",
        "frames": len(records),
        "hand_detected_frames": len(detected),
        "mediapipe_detected_frames": len(media),
        "hand_detection_rate": len(detected) / max(len(records), 1),
        "mediapipe_detection_rate": len(media) / max(len(records), 1),
        "candidate_pixels_total": base,
        "result_pixels_total": result,
        "retention_ratio": result / max(base, 1),
        "candidate_hand_overlap_pixels": candidate_hand_overlap,
        "result_hand_overlap_pixels": result_hand_overlap,
        "hand_overlap_reduction": 1.0 - result_hand_overlap / max(candidate_hand_overlap, 1),
        "elapsed_s": elapsed,
        "pipeline_fps": len(records) / max(elapsed, 1e-6),
        "search_radius": args.search_radius,
        "hand_dilate": args.hand_dilate,
        "min_area": args.min_area,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--images-dir")
    parser.add_argument("--image-glob", default="frame_*_raw.jpg")
    parser.add_argument("--candidate-mask-dir", required=True)
    parser.add_argument("--roi-mask-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--search-radius", type=int, default=220)
    parser.add_argument("--hand-dilate", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=450)
    parser.add_argument("--skin-fallback", action="store_true", help="Use conservative color fallback if landmarks miss.")
    parser.add_argument("--remove-yellow-overlay", action="store_true", help="Clean thin yellow contours in proxy input.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with mp.solutions.hands.Hands(
        static_image_mode=bool(args.images_dir),
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as hands:
        records = run_images(args, hands, out_dir) if args.images_dir else run_video(args, hands, out_dir)
    summary = summarize(records, time.perf_counter() - start, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
