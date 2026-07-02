#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def iter_pairs(root: Path, split: str):
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / image_path.name
        if mask_path.exists():
            yield image_path, mask_path


def copy_split(input_roots, output_root: Path, split: str):
    out_images = output_root / split / "images"
    out_masks = output_root / split / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    count = 0
    for root in input_roots:
        pairs = list(iter_pairs(root, split))
        prefix = root.name.replace(" ", "_")
        for image_path, mask_path in tqdm(pairs, desc=f"{split}:{root.name}"):
            stem = f"{prefix}_{count:07d}"
            shutil.copy2(image_path, out_images / f"{stem}{image_path.suffix.lower()}")
            shutil.copy2(mask_path, out_masks / f"{stem}.png")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_roots = [Path(path) for path in args.inputs]
    output_root = Path(args.output)
    counts = {split: copy_split(input_roots, output_root, split) for split in ("train", "val")}
    print(f"Merged datasets into {output_root}: {counts}")


if __name__ == "__main__":
    main()
