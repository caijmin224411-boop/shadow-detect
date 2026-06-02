#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def inspect_split(root: Path, split: str):
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    rows = []
    class_pixels = {0: 0, 1: 0, 2: 0}
    bad_masks = []
    if not image_dir.exists() or not mask_dir.exists():
        return {"exists": False}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            bad_masks.append(str(image_path.name))
            continue
        image = Image.open(image_path)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            if value in class_pixels:
                class_pixels[value] += int(count)
            else:
                bad_masks.append(f"{mask_path.name}: unexpected class {value}")
        rows.append({"image": image_path.name, "size": image.size})
    return {
        "exists": True,
        "pairs": len(rows),
        "class_pixels": class_pixels,
        "bad_masks": bad_masks[:20],
        "sample_sizes": rows[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root = Path(args.data)
    report = {"root": str(root), "train": inspect_split(root, "train"), "val": inspect_split(root, "val")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

