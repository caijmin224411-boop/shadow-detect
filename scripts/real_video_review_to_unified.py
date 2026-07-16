#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_rows(review_root: Path):
    manifest = review_root / "review_manifest.csv"
    if not manifest.exists():
        images = sorted((review_root / "images").glob("*"))
        return [{"stem": p.stem, "source_frame": i} for i, p in enumerate(images)]
    with manifest.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_name(idx: int, total: int, val_ratio: float, test_ratio: float) -> str:
    test_start = int(total * (1.0 - test_ratio))
    val_start = int(total * (1.0 - test_ratio - val_ratio))
    if idx >= test_start:
        return "test"
    if idx >= val_start:
        return "val"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--mask-dir", default="masks_reviewed")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Shuffle frames deterministically before assigning train/val/test splits.",
    )
    parser.add_argument(
        "--merge-nonpaper-into-background",
        action="store_true",
        help="Map label 3 to label 0 for exclusive three-class softmax training.",
    )
    args = parser.parse_args()

    review = Path(args.review)
    output = Path(args.output)
    rows = load_rows(review)
    if not rows:
        raise SystemExit(f"no review rows found in {review}")
    indexed_rows = list(enumerate(rows))
    if args.shuffle_seed is not None:
        np.random.default_rng(args.shuffle_seed).shuffle(indexed_rows)

    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "masks").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    class_pixels = {split: {0: 0, 1: 0, 2: 0, 3: 0} for split in counts}
    bad = []

    for split_idx, (source_idx, row) in enumerate(indexed_rows):
        stem = row["stem"]
        image_path = review / "images" / f"{stem}.png"
        mask_path = review / args.mask_dir / f"{stem}.png"
        if not image_path.exists() or not mask_path.exists():
            bad.append(stem)
            continue
        split = split_name(split_idx, len(rows), args.val_ratio, args.test_ratio)
        out_stem = f"real_video_v1_{source_idx:04d}"
        shutil.copy2(image_path, output / split / "images" / f"{out_stem}.png")
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        unexpected = sorted(int(v) for v in np.unique(mask) if int(v) not in {0, 1, 2, 3})
        if unexpected:
            bad.append(f"{stem}: unexpected labels {unexpected}")
            continue
        if args.merge_nonpaper_into_background:
            mask = mask.copy()
            mask[mask == 3] = 0
        Image.fromarray(mask).save(output / split / "masks" / f"{out_stem}.png")
        values, counts_px = np.unique(mask, return_counts=True)
        for value, count in zip(values, counts_px):
            class_pixels[split][int(value)] += int(count)
        counts[split] += 1

    summary = {"counts": counts, "class_pixels": class_pixels, "bad": bad[:20]}
    (output / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    if bad:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
