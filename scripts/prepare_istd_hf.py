#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def save_image(value, path: Path, mode: str):
    if isinstance(value, Image.Image):
        image = value.convert(mode)
    else:
        image = Image.open(value).convert(mode)
    image.save(path)


def save_mask(value, path: Path):
    if isinstance(value, Image.Image):
        mask = value.convert("L")
    else:
        mask = Image.open(value).convert("L")
    arr = np.asarray(mask)
    arr = (arr > 127).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Donghyun99/ISTD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="work/hf_cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-split", choices=("train", "val"), default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir).resolve()))
    from datasets import load_dataset

    resume_offset = 0
    if args.resume and args.target_split:
        resume_offset = len(list((output / args.target_split / "images").glob("*.png")))

    if args.no_streaming:
        ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
        split_names = list(ds.keys())
        source = ds[args.split] if args.split in ds else ds[split_names[0]]
        total = len(source) if args.limit <= 0 else min(args.limit, len(source))
        iterable = source.select(range(resume_offset, total))
    else:
        if args.limit <= 0:
            raise ValueError("Streaming mode requires --limit so full-dataset downloads are avoided")
        source = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir, streaming=True)
        total = args.limit
        iterable = source.skip(resume_offset).take(max(total - resume_offset, 0))
    val_start = int(total * (1.0 - args.val_ratio))

    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "masks").mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(enumerate(iterable, start=resume_offset), total=max(total - resume_offset, 0)):
        split = args.target_split or ("val" if idx >= val_start else "train")
        stem = f"istd_{idx:06d}"
        image_value = row.get("image") or row.get("shadow") or row.get("shadow_image")
        mask_value = row.get("mask") or row.get("shadow_mask")
        if image_value is None or mask_value is None:
            raise KeyError(f"Could not find image/mask columns in row keys: {sorted(row.keys())}")
        save_image(image_value, output / split / "images" / f"{stem}.png", "RGB")
        save_mask(mask_value, output / split / "masks" / f"{stem}.png")

    if not args.no_streaming:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
