#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def boxes_from_mask(mask: np.ndarray, min_area: int, max_boxes: int):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((area, [x, y, x + w, y + h]))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return np.asarray([box for _, box in boxes[:max_boxes]], dtype=np.float32)


def load_shadowsam(repo_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo_dir.resolve()))
    from sam import sam_model_registry  # type: ignore
    from sam.utils.transforms import ResizeLongestSide  # type: ignore

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device=device)
    model.eval()
    return model, ResizeLongestSide(model.image_encoder.img_size)


def predict_with_boxes(model, sam_trans, frame_rgb: np.ndarray, boxes: np.ndarray, device: torch.device):
    resize_image = sam_trans.apply_image(frame_rgb)
    image_tensor = torch.as_tensor(resize_image, device=device)
    input_image_torch = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image = model.preprocess(input_image_torch)
    original_image_size = frame_rgb.shape[:2]
    input_size = tuple(input_image_torch.shape[-2:])
    box = sam_trans.apply_boxes(boxes, original_image_size)
    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    with torch.no_grad():
        image_embedding = model.image_encoder(input_image)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
        upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        prob = torch.sigmoid(upscaled_masks)[0, 0].detach().float().cpu().numpy()
    return prob, iou_predictions.detach().float().cpu().numpy().tolist()


def postprocess(prob: np.ndarray, threshold: float, min_area: int):
    mask = (prob >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray, boxes: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(overlay, 0.82, fill, 0.18, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    for box in boxes.astype(int):
        x0, y0, x1, y1 = box.tolist()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 128, 255), 1)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--repo-dir", default="work/external/Detect-AnyShadow")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--prompt-min-area", type=int, default=2500)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()

    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    model, sam_trans = load_shadowsam(Path(args.repo_dir), Path(args.checkpoint), device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed_to_open={args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    records = []
    snapshots = {}
    processed = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        prompt_mask_path = Path(args.prompt_mask_dir) / f"frame_{processed:05d}.png"
        prompt_mask = cv2.imread(str(prompt_mask_path), cv2.IMREAD_GRAYSCALE)
        if prompt_mask is None:
            break
        boxes = boxes_from_mask(prompt_mask, args.prompt_min_area, args.max_boxes)
        if len(boxes) == 0:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            infer_ms = 0.0
            prob_mean = 0.0
            ious = []
        else:
            t0 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prob, ious = predict_with_boxes(model, sam_trans, frame_rgb, boxes, device)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            mask = postprocess(prob, args.threshold, args.min_area)
            prob_mean = float(prob.mean())
        overlay = draw_overlay(frame, mask, boxes)
        writer.write(overlay)
        cv2.imwrite(str(masks_dir / f"frame_{processed:05d}.png"), mask)
        if processed in {0, 1, 2, 3, 4, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = str(path.resolve())
        records.append((infer_ms, int(np.count_nonzero(mask)), prob_mean, len(boxes)))
        processed += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)
    summary = {
        "video": str(Path(args.video).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_mask_dir": str(Path(args.prompt_mask_dir).resolve()),
        "device": str(device),
        "threshold": args.threshold,
        "source_frames": source_frames,
        "processed_frames": processed,
        "elapsed_s": elapsed,
        "pipeline_fps": processed / max(elapsed, 1e-6),
        "infer_ms_mean": float(arr[:, 0].mean()) if len(arr) else 0.0,
        "mask_pixels_mean": float(arr[:, 1].mean()) if len(arr) else 0.0,
        "boxes_mean": float(arr[:, 3].mean()) if len(arr) else 0.0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
