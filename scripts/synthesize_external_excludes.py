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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def copy_dataset(input_root: Path, output_root: Path):
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            src = input_root / split / kind
            dst = output_root / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)


def find_external_pairs(root: Path):
    masks = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        text = "/".join(part.lower() for part in path.parts)
        if any(token in text for token in ("mask", "label", "segmentation", "annotation")):
            masks.append(path)
    return masks


def load_random_mask(mask_path: Path, target_shape, rng: np.random.Generator):
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return None
    h, w = target_shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = float(rng.uniform(0.18, 0.55))
    new_w = max(4, min(w - 1, int(w * scale)))
    new_h = max(4, min(h - 1, int(mask.shape[0] * (new_w / max(mask.shape[1], 1)))))
    if new_h >= h:
        new_h = max(4, h // 2)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    angle = float(rng.uniform(-35, 35))
    center = (new_w / 2.0, new_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)


def paste_exclude(image: np.ndarray, mask: np.ndarray, object_mask: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    mh, mw = object_mask.shape[:2]
    if mh >= h or mw >= w:
        return image, mask
    x0 = int(rng.integers(0, w - mw))
    y0 = int(rng.integers(0, h - mh))
    region = object_mask > 0
    alpha = float(rng.uniform(0.70, 0.96))
    color = int(rng.integers(12, 95))
    image2 = image.copy()
    patch = image2[y0:y0 + mh, x0:x0 + mw]
    patch[region] = np.clip(patch[region].astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    mask2 = mask.copy()
    mask_patch = mask2[y0:y0 + mh, x0:x0 + mw]
    mask_patch[region] = 2
    return image2, mask2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--objects", required=True, help="Folder containing hand/object masks from EgoHOS/EgoHands/COCO-like datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-objects", type=int, default=2000)
    parser.add_argument("--val-synthetic-objects", type=int, default=400)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    copy_dataset(input_root, output_root)

    external_masks = find_external_pairs(Path(args.objects))
    if not external_masks:
        raise RuntimeError(f"No external masks found in {args.objects}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    for split, count in (("train", args.synthetic_objects), ("val", args.val_synthetic_objects)):
        if count <= 0:
            continue
        split_images = sorted((input_root / split / "images").iterdir())
        out_images = output_root / split / "images"
        out_masks = output_root / split / "masks"
        for idx in tqdm(range(count), desc=f"{split} external excludes"):
            image_path = random.choice(split_images)
            mask_path = input_root / split / "masks" / f"{image_path.stem}.png"
            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            object_mask = None
            for _ in range(8):
                object_mask = load_random_mask(random.choice(external_masks), image.shape[:2], rng)
                if object_mask is not None:
                    break
            if object_mask is None:
                continue
            image2, mask2 = paste_exclude(image, mask, object_mask, rng)
            stem = f"external_exclude_{idx:07d}"
            Image.fromarray(image2).save(out_images / f"{stem}.png")
            Image.fromarray(mask2).save(out_masks / f"{stem}.png")


if __name__ == "__main__":
    main()
