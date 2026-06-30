#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from PIL import Image
from tqdm import tqdm


DATASET = "shadow-transfer-bench/ShadowTransfer"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def split_name(source_split: str) -> str:
    return "val" if source_split in {"val", "validation", "test"} else "train"


def collect_pairs(files, subset: str, resolution: str, source_split: str):
    prefix = f"{subset}/{resolution}/{source_split}/"
    images = [
        path for path in files
        if path.startswith(prefix + "images/") and Path(path).suffix.lower() in IMAGE_EXTS
    ]
    masks = {
        Path(path).stem: path
        for path in files
        if path.startswith(prefix + "masks/") and Path(path).suffix.lower() in IMAGE_EXTS
    }
    pairs = []
    for image_path in sorted(images):
        mask_path = masks.get(Path(image_path).stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def download_subset(subset: str, resolution: str, splits, cache_dir: str, max_workers: int):
    patterns = []
    for source_split in splits:
        base = f"{subset}/{resolution}/{source_split}"
        patterns.extend(
            [
                f"{base}/images/*",
                f"{base}/masks/*",
            ]
        )
    return Path(
        snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            allow_patterns=patterns,
            cache_dir=cache_dir,
            max_workers=max_workers,
        )
    )


def local_or_download(snapshot_root: Path | None, repo_path: str, cache_dir: str):
    if snapshot_root is not None:
        local = snapshot_root / repo_path
        if local.exists():
            return local
    return Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=repo_path,
            cache_dir=cache_dir,
        )
    )


def save_pair(
    image_repo_path: str,
    mask_repo_path: str,
    output_root: Path,
    target_split: str,
    stem: str,
    cache_dir: str,
    snapshot_root: Path | None,
):
    out_image = output_root / target_split / "images" / f"{stem}.png"
    out_mask = output_root / target_split / "masks" / f"{stem}.png"
    if out_image.exists() and out_mask.exists():
        return False
    image_local = local_or_download(snapshot_root, image_repo_path, cache_dir)
    mask_local = local_or_download(snapshot_root, mask_repo_path, cache_dir)
    Image.open(image_local).convert("RGB").save(out_image)
    mask = Image.open(mask_local).convert("L")
    arr = (np.asarray(mask) > 127).astype(np.uint8)
    Image.fromarray(arr).save(out_mask)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="work/hf_cache_shadowtransfer")
    parser.add_argument(
        "--subset",
        default="data_loco/fold_2_holdout_chicago",
        help="Repo folder to read, e.g. data_loco/fold_2_holdout_chicago or data_cities/chicago.",
    )
    parser.add_argument("--resolution", default="midres", choices=("midres", "highres", "lowres"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--no-snapshot", action="store_true", help="Download files one by one instead of prefetching the subset.")
    parser.add_argument("--snapshot-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir).resolve()))
    output = Path(args.output)
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "masks").mkdir(parents=True, exist_ok=True)

    files = list_repo_files(DATASET, repo_type="dataset")
    rng = random.Random(args.seed)
    snapshot_root = None
    source_splits = [args.train_split, args.val_split]
    if not args.no_snapshot:
        try:
            snapshot_root = download_subset(
                args.subset,
                args.resolution,
                source_splits,
                args.cache_dir,
                args.snapshot_workers,
            )
        except Exception as exc:
            print(f"WARNING: snapshot prefetch failed, falling back to per-file downloads: {exc}")

    written = {"train": 0, "val": 0}
    skipped = {"train": 0, "val": 0}
    for source_split, limit in ((args.train_split, args.train_limit), (args.val_split, args.val_limit)):
        pairs = collect_pairs(files, args.subset, args.resolution, source_split)
        if not pairs:
            raise RuntimeError(f"No image/mask pairs found under {args.subset}/{args.resolution}/{source_split}")
        rng.shuffle(pairs)
        if limit > 0:
            pairs = pairs[:limit]
        target_split = split_name(source_split)
        if source_split == args.val_split:
            target_split = "val"
        for idx, (image_path, mask_path) in tqdm(enumerate(pairs), total=len(pairs), desc=f"{source_split}->{target_split}"):
            stem = f"shadowtransfer_{target_split}_{idx:07d}"
            if save_pair(image_path, mask_path, output, target_split, stem, args.cache_dir, snapshot_root):
                written[target_split] += 1
            else:
                skipped[target_split] += 1
    print(f"Wrote ShadowTransfer unified dataset to {output}: written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
