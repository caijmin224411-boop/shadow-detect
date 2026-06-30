#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


LABEL_NAMES = {
    0: "clean/background",
    1: "paper_shadow",
    2: "hand/object/arm_exclude",
    3: "non_paper_optional",
}

LABEL_COLORS_BGR = {
    1: np.array([0, 255, 255], dtype=np.uint8),
    2: np.array([0, 80, 255], dtype=np.uint8),
    3: np.array([255, 0, 255], dtype=np.uint8),
}


def read_manifest(review_dir: Path) -> list[str]:
    manifest = review_dir / "review_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        names = [row[key] for row in rows for key in ("name", "stem") if row.get(key)]
        if names:
            return names
    image_dir = review_dir / "images"
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def load_image_and_mask(review_dir: Path, name: str):
    image_path = None
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = review_dir / "images" / f"{name}{suffix}"
        if candidate.exists():
            image_path = candidate
            break
    if image_path is None:
        image_path = review_dir / "images" / f"{name}.png"
    mask_path = review_dir / "masks_reviewed" / f"{name}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 3, 1, mask).astype(np.uint8)
    return image, mask, mask_path


def make_view(image: np.ndarray, mask: np.ndarray, label: int, brush: int, name: str, index: int, total: int):
    overlay = image.copy()
    for cls, color in LABEL_COLORS_BGR.items():
        overlay[mask == cls] = (overlay[mask == cls].astype(np.float32) * 0.45 + color.astype(np.float32) * 0.55).astype(
            np.uint8
        )
    contours_shadow, _ = cv2.findContours((mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_exclude, _ = cv2.findContours((mask == 2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_shadow, -1, (0, 255, 255), 2)
    cv2.drawContours(overlay, contours_exclude, -1, (0, 80, 255), 2)

    info = f"{index + 1}/{total} {name} | label {label}: {LABEL_NAMES[label]} | brush {brush}px"
    help_line = "keys: 0 erase, 1 shadow, 2 exclude, 3 non-paper, +/- brush, n next, p prev, s save, q quit"
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 54), (20, 20, 20), -1)
    cv2.putText(overlay, info, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, help_line, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return overlay


def save_mask(mask_path: Path, mask: np.ndarray):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(mask_path), mask.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"failed to write {mask_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="work/datasets/real_video_v1_review")
    parser.add_argument("--start", default="")
    parser.add_argument("--brush", type=int, default=18)
    parser.add_argument("--window-scale", type=float, default=0.85)
    args = parser.parse_args()

    review_dir = Path(args.review)
    names = read_manifest(review_dir)
    if not names:
        print(f"no review frames found under {review_dir}")
        return 2

    index = names.index(args.start) if args.start in names else 0
    label = 1
    brush = max(1, args.brush)
    drawing = False
    dirty = False
    image, mask, mask_path = load_image_and_mask(review_dir, names[index])
    window_name = "real video mask review"

    def paint(event, x, y, flags, _param):
        nonlocal drawing, dirty, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        if drawing or (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(mask, (x, y), brush, int(label), -1)
            dirty = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, paint)

    while True:
        view = make_view(image, mask, label, brush, names[index], index, len(names))
        if args.window_scale > 0:
            cv2.resizeWindow(
                window_name,
                int(view.shape[1] * args.window_scale),
                int(view.shape[0] * args.window_scale),
            )
        cv2.imshow(window_name, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            if dirty:
                save_mask(mask_path, mask)
                print(f"saved {mask_path}")
            break
        if key in (ord("s"),):
            save_mask(mask_path, mask)
            dirty = False
            print(f"saved {mask_path}")
        elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
            label = int(chr(key))
            print(f"label={label} {LABEL_NAMES[label]}")
        elif key in (ord("+"), ord("=")):
            brush = min(160, brush + 3)
        elif key in (ord("-"), ord("_")):
            brush = max(1, brush - 3)
        elif key in (ord("n"), ord(" "), 83):
            if dirty:
                save_mask(mask_path, mask)
                print(f"saved {mask_path}")
            index = min(len(names) - 1, index + 1)
            image, mask, mask_path = load_image_and_mask(review_dir, names[index])
            dirty = False
        elif key in (ord("p"), 81):
            if dirty:
                save_mask(mask_path, mask)
                print(f"saved {mask_path}")
            index = max(0, index - 1)
            image, mask, mask_path = load_image_and_mask(review_dir, names[index])
            dirty = False

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
