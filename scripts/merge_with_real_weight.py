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
        if mask_path.exists():
            yield image_path, mask_path


def copy_pairs(root: Path, output: Path, split: str, prefix: str, repeat: int, start_count: int) -> int:
    out_images = output / split / "images"
    out_masks = output / split / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    pairs = list(iter_pairs(root, split))
    count = start_count
    for rep in range(repeat):
        for image_path, mask_path in tqdm(pairs, desc=f"{split}:{prefix}:x{rep + 1}"):
            stem = f"{prefix}_{count:07d}"
            shutil.copy2(image_path, out_images / f"{stem}{image_path.suffix.lower()}")
            shutil.copy2(mask_path, out_masks / f"{stem}.png")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--real-repeat", type=int, default=8)
    args = parser.parse_args()

    public = Path(args.public)
    real = Path(args.real)
    output = Path(args.output)
    summary = {}
    for split in ("train", "val"):
        count = 0
        count = copy_pairs(public, output, split, "public", 1, count)
        real_repeat = args.real_repeat if split == "train" else 1
        count = copy_pairs(real, output, split, "real", real_repeat, count)
        summary[split] = count
    print(f"Merged weighted dataset into {output}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
