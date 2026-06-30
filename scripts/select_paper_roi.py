#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_points(values: list[str]) -> list[tuple[int, int]]:
    points = []
    for value in values:
        x_str, y_str = value.split(",", 1)
        points.append((int(float(x_str)), int(float(y_str))))
    return points


def make_mask(size: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def draw_preview(frame: np.ndarray, points: list[tuple[int, int]], mask: np.ndarray | None = None) -> np.ndarray:
    preview = frame.copy()
    if mask is not None:
        fill = np.zeros_like(frame)
        fill[mask > 0] = (0, 180, 255)
        preview = cv2.addWeighted(preview, 0.78, fill, 0.22, 0)
    for idx, point in enumerate(points):
        cv2.circle(preview, point, 5, (0, 255, 255), -1)
        cv2.putText(preview, str(idx + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if len(points) >= 2:
        cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], len(points) >= 3, (0, 255, 255), 2)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 56), (20, 20, 20), -1)
    cv2.putText(preview, "click paper corners | u undo | r reset | s save | q quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, "Tip: include all paper, exclude table/background if possible", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
    return preview


def first_frame(video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read first frame from {video}")
    return frame


def interactive_points(frame: np.ndarray, window_scale: float) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    window = "select paper roi"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        mask = make_mask(frame.shape[:2], points) if len(points) >= 3 else None
        preview = draw_preview(frame, points, mask)
        if window_scale > 0:
            cv2.resizeWindow(window, int(frame.shape[1] * window_scale), int(frame.shape[0] * window_scale))
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            points = []
            break
        if key in (ord("u"), 8, 127):
            if points:
                points.pop()
        elif key == ord("r"):
            points = []
        elif key in (ord("s"), 13):
            if len(points) >= 3:
                break
            print("Need at least 3 points.")
    cv2.destroyAllWindows()
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default="work/datasets/real_video_v1_paper_roi")
    parser.add_argument("--point", action="append", default=[], help="x,y. Repeat this for polygon vertices.")
    parser.add_argument("--window-scale", type=float, default=0.75)
    args = parser.parse_args()

    frame = first_frame(Path(args.video))
    points = parse_points(args.point) if args.point else interactive_points(frame, args.window_scale)
    if len(points) < 3:
        print("No ROI saved.")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = make_mask(frame.shape[:2], points)
    preview = draw_preview(frame, points, mask)
    cv2.imwrite(str(out_dir / "paper_roi_mask.png"), mask)
    cv2.imwrite(str(out_dir / "paper_roi_preview.jpg"), preview)
    (out_dir / "paper_roi.json").write_text(
        json.dumps({"video": str(Path(args.video).resolve()), "points": points, "size": [frame.shape[1], frame.shape[0]]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"mask": str((out_dir / "paper_roi_mask.png").resolve()), "preview": str((out_dir / "paper_roi_preview.jpg").resolve()), "points": points}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
