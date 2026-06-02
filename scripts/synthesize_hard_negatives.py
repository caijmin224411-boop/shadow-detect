#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def copy_dataset(input_root: Path, output_root: Path):
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            src = input_root / split / kind
            dst = output_root / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)


def draw_dark_object(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    overlay = image.copy()
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    count = int(rng.integers(1, 4))
    for _ in range(count):
        kind = rng.choice(["rect", "ellipse", "line"])
        color = int(rng.integers(15, 90))
        if kind == "rect":
            x0, y0 = int(rng.integers(0, w - 8)), int(rng.integers(0, h - 8))
            x1 = min(w - 1, x0 + int(rng.integers(w // 12, w // 3)))
            y1 = min(h - 1, y0 + int(rng.integers(h // 18, h // 4)))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(obj_mask, (x0, y0), (x1, y1), 255, -1)
        elif kind == "ellipse":
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(w // 12, w // 4)), int(rng.integers(h // 20, h // 6)))
            angle = float(rng.integers(0, 180))
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
            cv2.ellipse(obj_mask, center, axes, angle, 0, 360, 255, -1)
        else:
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            thickness = int(rng.integers(4, 16))
            cv2.line(overlay, p0, p1, color, thickness)
            cv2.line(obj_mask, p0, p1, 255, thickness)
    alpha = float(rng.uniform(0.65, 0.95))
    image2 = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    mask2 = mask.copy()
    mask2[obj_mask > 0] = 2
    return image2, mask2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-objects", type=int, default=1000)
    parser.add_argument("--val-synthetic-objects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    copy_dataset(input_root, output_root)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    for split, count in (("train", args.synthetic_objects), ("val", args.val_synthetic_objects)):
        if count <= 0:
            continue
        split_images = sorted((input_root / split / "images").iterdir())
        if not split_images:
            raise RuntimeError(f"No {split} images found")

        out_images = output_root / split / "images"
        out_masks = output_root / split / "masks"
        for idx in tqdm(range(count), desc=f"{split} hard negatives"):
            image_path = random.choice(split_images)
            mask_path = input_root / split / "masks" / f"{image_path.stem}.png"
            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            image2, mask2 = draw_dark_object(image, mask, rng)
            stem = f"hardneg_{idx:06d}"
            Image.fromarray(image2).save(out_images / f"{stem}.png")
            Image.fromarray(mask2).save(out_masks / f"{stem}.png")


if __name__ == "__main__":
    main()
