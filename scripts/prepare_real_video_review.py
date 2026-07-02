#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["shadow_logits"], outputs["exclude_logits"], outputs["paper_logits"]
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        by_name = {getattr(t, "name", "").split("/")[0]: t for t in outputs}
        if {"shadow_logits", "exclude_logits", "paper_logits"} <= set(by_name):
            return by_name["shadow_logits"], by_name["exclude_logits"], by_name["paper_logits"]
        return outputs[0], outputs[1], outputs[2]
    raise TypeError(f"Expected three model outputs, got {type(outputs)!r}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def local_shadow_score(gray48: np.ndarray) -> np.ndarray:
    gray_f = gray48.astype(np.float32)
    mean = cv2.blur(gray_f, (9, 9), borderType=cv2.BORDER_REPLICATE)
    score = np.clip((mean - gray_f - 5.0) / 45.0, 0.0, 1.0)
    return score


def edge_score(gray48: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray48, 40, 110)
    return cv2.blur((edges > 0).astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)


def make_input(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 127.5 - 1.0
    return x[None, ..., None], gray


def infer_draft_mask(model, frame_bgr: np.ndarray, args) -> tuple[np.ndarray, dict]:
    x, gray = make_input(frame_bgr)
    shadow_logits, exclude_logits, paper_logits = unpack_outputs(model(x, training=False))
    shadow_prob = sigmoid(shadow_logits.numpy()[0, ..., 0])
    exclude_prob = sigmoid(exclude_logits.numpy()[0, ..., 0])
    paper_prob = sigmoid(paper_logits.numpy()[0, ..., 0])

    gray48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    dark_score = local_shadow_score(gray48)
    edge = edge_score(gray48)

    shadow = (
        (shadow_prob >= args.shadow_threshold)
        | ((shadow_prob >= args.shadow_soft_threshold) & (dark_score >= args.dark_score_threshold))
    )
    exclude = (exclude_prob >= args.exclude_threshold) | ((edge >= args.edge_exclude_threshold) & (gray48 < args.dark_object_gray))
    paper = paper_prob >= args.paper_threshold

    mask48 = np.zeros((48, 48), dtype=np.uint8)
    mask48[shadow & paper & ~exclude] = 1
    mask48[exclude] = 2

    kernel = np.ones((3, 3), np.uint8)
    shadow_only = (mask48 == 1).astype(np.uint8)
    shadow_only = cv2.morphologyEx(shadow_only, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask48[(shadow_only > 0) & (mask48 != 2)] = 1

    full = cv2.resize(mask48, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    stats = {
        "shadow_ratio": float(np.mean(mask48 == 1)),
        "exclude_ratio": float(np.mean(mask48 == 2)),
        "shadow_prob_mean": float(np.mean(shadow_prob)),
        "exclude_prob_mean": float(np.mean(exclude_prob)),
        "paper_prob_mean": float(np.mean(paper_prob)),
    }
    return full, stats


def draw_preview(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = frame_bgr.copy()
    shadow = (mask == 1).astype(np.uint8) * 255
    exclude = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
    contours, _ = cv2.findContours(exclude, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 0, 255), 2)
    return preview


def pick_frame_indices(frame_count: int, target: int) -> list[int]:
    if frame_count <= target:
        return list(range(frame_count))
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, target)))


def write_contact_sheet(preview_paths: list[Path], out_path: Path, thumb_w: int = 240):
    thumbs = []
    for path in preview_paths:
        image = Image.open(path).convert("RGB")
        thumb_h = max(1, int(image.height * (thumb_w / image.width)))
        image = image.resize((thumb_w, thumb_h))
        thumbs.append((path.stem, image))
    if not thumbs:
        return
    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(img.height for _, img in thumbs) + 24
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * cell_h
        sheet.paste(img, (x, y + 18))
        draw.text((x + 4, y + 2), label, fill=(0, 0, 0))
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="work/datasets/real_video_v1_review")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--shadow-soft-threshold", type=float, default=0.25)
    parser.add_argument("--exclude-threshold", type=float, default=0.35)
    parser.add_argument("--paper-threshold", type=float, default=0.10)
    parser.add_argument("--dark-score-threshold", type=float, default=0.22)
    parser.add_argument("--edge-exclude-threshold", type=float, default=0.28)
    parser.add_argument("--dark-object-gray", type=int, default=80)
    args = parser.parse_args()

    out = Path(args.output)
    for subdir in ("images", "masks_draft", "masks_reviewed", "previews"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {args.video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(pick_frame_indices(frame_count, args.frames))
    model = tf.keras.models.load_model(args.model, compile=False)

    rows = []
    preview_paths = []
    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in indices:
            frame_idx += 1
            continue
        stem = f"realv1_{kept:04d}_f{frame_idx:06d}"
        mask, stats = infer_draft_mask(model, frame, args)
        image_path = out / "images" / f"{stem}.png"
        draft_path = out / "masks_draft" / f"{stem}.png"
        reviewed_path = out / "masks_reviewed" / f"{stem}.png"
        preview_path = out / "previews" / f"{stem}.jpg"
        cv2.imwrite(str(image_path), frame)
        Image.fromarray(mask.astype(np.uint8)).save(draft_path)
        Image.fromarray(mask.astype(np.uint8)).save(reviewed_path)
        cv2.imwrite(str(preview_path), draw_preview(frame, mask))
        rows.append(
            {
                "stem": stem,
                "source_frame": frame_idx,
                "image": str(image_path),
                "draft_mask": str(draft_path),
                "reviewed_mask": str(reviewed_path),
                "preview": str(preview_path),
                **stats,
                "review_status": "draft_copied_to_reviewed",
            }
        )
        preview_paths.append(preview_path)
        kept += 1
        frame_idx += 1
    cap.release()

    with (out / "review_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["stem"])
        writer.writeheader()
        writer.writerows(rows)
    write_contact_sheet(preview_paths, out / "contact_sheet.jpg")
    (out / "README.md").write_text(
        "Review masks in masks_reviewed. Label 1 is shadow, label 2 is hand/object/arm exclude. "
        "Draft masks were copied there so the dataset can be converted immediately, but manual review is recommended.\n",
        encoding="utf-8",
    )
    print(f"Wrote {kept} review frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
