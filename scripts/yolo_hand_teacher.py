#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "This optional teacher script needs ultralytics. Install it with: "
            "pip install ultralytics"
        ) from exc
    return YOLO


def parse_names(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def copy_split(input_root: Path, output_root: Path, split: str):
    for subdir in ("images", "masks"):
        src_dir = input_root / split / subdir
        dst_dir = output_root / split / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)


def draw_detection_mask(result, image_shape, class_names, confidence: float, dilate_px: int):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return mask

    selected = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        cls_name = str(names.get(cls_id, cls_id)).lower()
        score = float(box.conf.item())
        if score >= confidence and cls_name in class_names:
            selected.append(i)

    result_masks = getattr(result, "masks", None)
    if result_masks is not None and result_masks.data is not None:
        masks = result_masks.data.cpu().numpy()
        for i in selected:
            resized = cv2.resize(masks[i].astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask[resized > 0] = 255
    else:
        for i in selected:
            xyxy = boxes[i].xyxy.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(image_shape[1] - 1, x1))
            x2 = max(0, min(image_shape[1] - 1, x2))
            y1 = max(0, min(image_shape[0] - 1, y1))
            y2 = max(0, min(image_shape[0] - 1, y2))
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255

    if dilate_px > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def process_split(model, input_root: Path, output_root: Path, split: str, class_names, confidence: float, dilate_px: int):
    image_dir = input_root / split / "images"
    mask_dir = output_root / split / "masks"
    count = 0
    changed = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        out_mask_path = mask_dir / f"{image_path.stem}.png"
        if not out_mask_path.exists():
            out_mask_path = mask_dir / image_path.name
        image = np.array(Image.open(image_path).convert("RGB"))
        results = model.predict(image, verbose=False)
        teacher_mask = draw_detection_mask(results[0], image.shape, class_names, confidence, dilate_px)
        if teacher_mask.any():
            mask = np.array(Image.open(out_mask_path))
            mask[teacher_mask > 0] = 2
            Image.fromarray(mask.astype(np.uint8)).save(out_mask_path)
            changed += 1
        count += 1
    return count, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Ultralytics model path, HF path, or URL accepted by YOLO().")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--class-names", default="hand,hands")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--dilate-px", type=int, default=8)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    YOLO = ensure_ultralytics()
    model = YOLO(args.model)
    class_names = parse_names(args.class_names)

    for split in parse_names(args.splits):
        copy_split(input_root, output_root, split)
        total, changed = process_split(model, input_root, output_root, split, class_names, args.confidence, args.dilate_px)
        print(f"{split}: scanned {total} images, added YOLO teacher exclusions to {changed}")


if __name__ == "__main__":
    main()
