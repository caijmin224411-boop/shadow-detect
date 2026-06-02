#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_sample(rng: np.random.Generator, h: int = 192, w: int = 192):
    base = int(rng.integers(180, 245))
    image = np.full((h, w), base, dtype=np.float32)
    image += rng.normal(0, 5, (h, w))

    yy, xx = np.mgrid[0:h, 0:w]
    vignette = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * h)
    image -= vignette * rng.uniform(8, 28)

    mask = np.zeros((h, w), dtype=np.uint8)
    shadow = np.zeros((h, w), dtype=np.uint8)
    center = (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20)))
    axes = (int(rng.integers(w // 5, w // 2)), int(rng.integers(h // 8, h // 3)))
    angle = float(rng.integers(0, 180))
    cv2.ellipse(shadow, center, axes, angle, 0, 360, 255, -1)
    shadow = cv2.GaussianBlur(shadow, (31, 31), 0)
    shadow_strength = rng.uniform(35, 95)
    image -= (shadow.astype(np.float32) / 255.0) * shadow_strength
    mask[shadow > 50] = 1

    # Draw printed/handwritten marks that should remain paper/background.
    for _ in range(int(rng.integers(4, 16))):
        y = int(rng.integers(10, h - 10))
        x0 = int(rng.integers(8, w // 2))
        x1 = int(rng.integers(w // 2, w - 8))
        color = int(rng.integers(70, 145))
        cv2.line(image, (x0, y), (x1, y + int(rng.integers(-2, 3))), color, 1)

    # Draw dark hard negatives: hands/objects are class 2, not shadow.
    obj = np.zeros((h, w), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 3))):
        kind = rng.choice(["ellipse", "rect", "line"])
        if kind == "ellipse":
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            a = (int(rng.integers(w // 12, w // 4)), int(rng.integers(h // 12, h // 4)))
            cv2.ellipse(obj, c, a, float(rng.integers(0, 180)), 0, 360, 255, -1)
        elif kind == "rect":
            x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
            x1, y1 = min(w - 1, x0 + int(rng.integers(12, 50))), min(h - 1, y0 + int(rng.integers(12, 50)))
            cv2.rectangle(obj, (x0, y0), (x1, y1), 255, -1)
        else:
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(obj, p0, p1, 255, int(rng.integers(5, 18)))
    image[obj > 0] = rng.uniform(25, 95)
    mask[obj > 0] = 2

    return np.clip(image, 0, 255).astype(np.uint8), mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.output)
    rng = np.random.default_rng(args.seed)
    for split, count in (("train", args.train), ("val", args.val)):
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image, mask = make_sample(rng)
            stem = f"synthetic_{idx:06d}"
            Image.fromarray(image).save(image_dir / f"{stem}.png")
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
