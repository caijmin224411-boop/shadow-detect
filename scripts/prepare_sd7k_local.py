#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_HINTS = ("shadow", "image", "input", "img")
MASK_HINTS = ("mask", "gt", "label")


def find_files(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def score_path(path: Path, hints):
    text = "/".join(part.lower() for part in path.parts)
    return sum(1 for hint in hints if hint in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    files = find_files(source)
    masks = [p for p in files if score_path(p, MASK_HINTS) > 0]
    images = [p for p in files if score_path(p, IMAGE_HINTS) > 0 and score_path(p, MASK_HINTS) == 0]
    by_stem = {p.stem: p for p in images}

    pairs = []
    for mask in masks:
        image = by_stem.get(mask.stem)
        if image is not None:
            pairs.append((image, mask))

    if not pairs:
        raise RuntimeError("No SD7K image/mask pairs found. Pass a folder with matching image and mask stems.")

    random.Random(args.seed).shuffle(pairs)
    val_start = int(len(pairs) * (1.0 - args.val_ratio))

    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "masks").mkdir(parents=True, exist_ok=True)

    for idx, (image_path, mask_path) in tqdm(enumerate(pairs), total=len(pairs)):
        split = "val" if idx >= val_start else "train"
        stem = f"sd7k_{idx:06d}"
        Image.open(image_path).convert("RGB").save(output / split / "images" / f"{stem}.png")
        mask = Image.open(mask_path).convert("L")
        arr = (np.asarray(mask) > 127).astype(np.uint8)
        Image.fromarray(arr).save(output / split / "masks" / f"{stem}.png")


if __name__ == "__main__":
    main()
