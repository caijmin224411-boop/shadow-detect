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


def choose_device(name: str) -> torch.device:
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def load_predictor(repo_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo_dir.resolve()))
    from sam import SamPredictor, sam_model_registry  # type: ignore

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def mask_to_boxes(mask: np.ndarray, max_boxes: int, min_area: int, pad: int) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = mask.shape[:2]
    boxes: list[tuple[int, np.ndarray]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        box = np.array(
            [
                max(0, x - pad),
                max(0, y - pad),
                min(w - 1, x + bw + pad),
                min(h - 1, y + bh + pad),
            ],
            dtype=np.float32,
        )
        boxes.append((area, box))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:max_boxes]]


def merge_sam_masks(predictor, boxes: list[np.ndarray], proposal: np.ndarray) -> np.ndarray:
    if not boxes:
        return np.zeros_like(proposal, dtype=np.uint8)
    merged = np.zeros_like(proposal, dtype=np.uint8)
    proposal_bool = proposal > 0
    for box in boxes:
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True,
            return_logits=False,
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
        # Keep SAM from grabbing unrelated objects far outside the original proposal.
        dilated = cv2.dilate(proposal_bool.astype(np.uint8), np.ones((31, 31), np.uint8), iterations=1) > 0
        merged[mask & dilated] = 255
    return merged


def postprocess(mask: np.ndarray, min_area: int) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(overlay, 0.82, fill, 0.18, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def proposal_path_for(proposal_dir: Path, processed_index: int, read_index: int) -> Path:
    candidates = [
        proposal_dir / f"frame_{processed_index:05d}.png",
        proposal_dir / f"frame_{read_index:05d}.png",
        proposal_dir / f"{processed_index:05d}.png",
        proposal_dir / f"{read_index:05d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--repo-dir", default="work/external/Detect-AnyShadow")
    parser.add_argument("--checkpoint", default="work/external/Detect-AnyShadow/checkpoints/chk_sam/finetune.pth")
    parser.add_argument("--proposal-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-boxes", type=int, default=4)
    parser.add_argument("--box-min-area", type=int, default=1200)
    parser.add_argument("--box-pad", type=int, default=18)
    parser.add_argument("--min-area", type=int, default=900)
    args = parser.parse_args()

    device = choose_device(args.device)
    predictor = load_predictor(Path(args.repo_dir), Path(args.checkpoint), device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed_to_open={args.video}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (width, height),
    )

    records = []
    snapshots = {}
    processed = 0
    read_index = 0
    start = time.perf_counter()
    proposal_dir = Path(args.proposal_mask_dir)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % max(1, args.frame_stride) != 0:
            read_index += 1
            continue

        prop_path = proposal_path_for(proposal_dir, processed, read_index)
        proposal = cv2.imread(str(prop_path), cv2.IMREAD_GRAYSCALE)
        if proposal is None:
            proposal = np.zeros((height, width), dtype=np.uint8)
        elif proposal.shape[:2] != (height, width):
            proposal = cv2.resize(proposal, (width, height), interpolation=cv2.INTER_NEAREST)

        t0 = time.perf_counter()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        boxes = mask_to_boxes(proposal, args.max_boxes, args.box_min_area, args.box_pad)
        mask = merge_sam_masks(predictor, boxes, proposal)
        mask = postprocess(mask, args.min_area)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        overlay = draw_overlay(frame, mask)
        writer.write(overlay)
        cv2.imwrite(str(masks_dir / f"frame_{processed:05d}.png"), mask)
        if processed in {0, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = str(path.resolve())
        records.append((infer_ms, int(np.count_nonzero(mask)), len(boxes)))

        processed += 1
        read_index += 1
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    arr = np.asarray(records, dtype=np.float32)
    summary = {
        "video": str(Path(args.video).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "proposal_mask_dir": str(proposal_dir.resolve()),
        "device": str(device),
        "source_frames": source_frames,
        "source_fps": fps,
        "size": [width, height],
        "processed_frames": processed,
        "elapsed_s": elapsed,
        "pipeline_fps": processed / max(elapsed, 1e-6),
        "infer_ms_mean": float(arr[:, 0].mean()) if len(arr) else 0.0,
        "infer_ms_p95": float(np.percentile(arr[:, 0], 95)) if len(arr) else 0.0,
        "mask_pixels_mean": float(arr[:, 1].mean()) if len(arr) else 0.0,
        "mask_pixels_max": int(arr[:, 1].max()) if len(arr) else 0,
        "boxes_mean": float(arr[:, 2].mean()) if len(arr) else 0.0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
