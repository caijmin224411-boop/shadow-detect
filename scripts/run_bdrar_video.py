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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_bdrar_model(repo_dir: Path, weight_path: Path, device: torch.device):
    sys.path.insert(0, str(repo_dir.resolve()))
    from model import BDRAR  # type: ignore

    model = BDRAR().to(device)
    state = torch.load(str(weight_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_frame(frame_bgr: np.ndarray, input_size: int):
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def infer_mask(model, frame_bgr: np.ndarray, device: torch.device, input_size: int):
    x = preprocess_frame(frame_bgr, input_size).to(device)
    with torch.no_grad():
        pred = model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = F.interpolate(pred, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
        prob = pred.squeeze().detach().float().cpu().numpy()
    return np.clip(prob, 0.0, 1.0)


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


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(overlay, 0.82, fill, 0.18, 0)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--repo-dir", default="work/external/Shadow-Detection-and-Removal")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
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
    model = load_bdrar_model(Path(args.repo_dir), Path(args.weights), device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed_to_open={args.video}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % max(1, args.frame_stride) != 0:
            read_index += 1
            continue

        t0 = time.perf_counter()
        prob = infer_mask(model, frame, device, args.input_size)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        mask = postprocess(prob, args.threshold, args.min_area)
        overlay = draw_overlay(frame, mask)
        writer.write(overlay)
        cv2.imwrite(str(masks_dir / f"frame_{processed:05d}.png"), mask)
        if processed in {0, 30, 90, 150, 240, 360}:
            path = out_dir / f"frame_{processed:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[processed] = str(path.resolve())
        records.append((infer_ms, int(np.count_nonzero(mask)), float(prob.mean()), float(prob.max())))
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
        "weights": str(Path(args.weights).resolve()),
        "device": str(device),
        "input_size": args.input_size,
        "threshold": args.threshold,
        "min_area": args.min_area,
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
        "prob_mean_last": float(records[-1][2]) if records else 0.0,
        "prob_max_last": float(records[-1][3]) if records else 0.0,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
