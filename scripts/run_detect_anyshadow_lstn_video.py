#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


class _TimerEvent:
    def __init__(self, enable_timing: bool = True):
        self.t = 0.0

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, other: "_TimerEvent") -> float:
        return max(0.0, (other.t - self.t) * 1000.0)


def patch_cuda_to_cpu() -> None:
    torch.cuda.set_device = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.cuda.empty_cache = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.cuda.synchronize = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0  # type: ignore[assignment]
    torch.cuda.Event = _TimerEvent  # type: ignore[assignment]
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]

    original_load = torch.load

    def load_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    torch.load = load_cpu  # type: ignore[assignment]


def load_frames(video: Path, max_frames: int, frame_stride: int) -> tuple[list[np.ndarray], float, tuple[int, int], int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    read_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % max(1, frame_stride) == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        read_index += 1
    cap.release()
    if not frames:
        raise RuntimeError("no frames loaded")
    h, w = frames[0].shape[:2]
    return frames, fps / max(1, frame_stride), (w, h), source_frames


def read_seed_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read seed mask {path}")
    w, h = size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)


def draw_overlay(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(frame_bgr, 0.82, fill, 0.18, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--repo-dir", default="work/external/Detect-AnyShadow")
    parser.add_argument("--seed-mask", required=True, help="First-frame binary shadow mask.")
    parser.add_argument("--ckpt-path", default="work/external/Detect-AnyShadow/checkpoints/lstnb")
    parser.add_argument("--model", default="lstnb", choices=["lstnt", "lstns", "lstnb"])
    parser.add_argument("--start-step", type=int, default=10000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-resolution", type=float, default=240.0)
    args = parser.parse_args()

    patch_cuda_to_cpu()
    repo_dir = Path(args.repo_dir).resolve()
    video_path = Path(args.video).resolve()
    seed_mask_path = Path(args.seed_mask).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    sys.path.insert(0, str((repo_dir / "lstn").resolve()))
    sys.path.insert(0, str(repo_dir.resolve()))
    cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        from lstn.configs.visha import EngineConfig  # type: ignore
        from lstn.networks.managers.eval_demo import Evaluator  # type: ignore

        frames, fps, size, source_frames = load_frames(video_path, args.max_frames, args.frame_stride)
        seed = read_seed_mask(seed_mask_path, size)
        cfg = EngineConfig("lstn", args.model)
        cfg.TEST_EMA = False
        cfg.TEST_CKPT_STEP = args.start_step
        cfg.TEST_CKPT_PATH = str(ckpt_path)
        cfg.TEST_DATASET_PATH = ""
        cfg.TEST_MAX_LONG_EDGE = args.max_resolution * 800.0 / 480.0
        cfg.TEST_WORKERS = 0
        cfg.TEST_FRAME_LOG = False
        cfg.MODEL_LSAB_ENABLE_CORR = False

        start = time.perf_counter()
        evaluator = Evaluator(cfg, frames, [seed])
        propagated, log, log_time = evaluator.evaluating()
        elapsed = time.perf_counter() - start
    finally:
        os.chdir(cwd)

    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    all_masks = [seed] + propagated
    snapshots = {}
    records = []
    for idx, (frame, mask) in enumerate(zip(frames, all_masks)):
        cv2.imwrite(str(masks_dir / f"frame_{idx:05d}.png"), mask)
        overlay = draw_overlay(frame, mask)
        writer.write(overlay)
        if idx in {0, 1, 10, 30, 50, 90, 120, 150, 240, 300, 360, 420}:
            path = out_dir / f"frame_{idx:04d}_overlay.jpg"
            cv2.imwrite(str(path), overlay)
            snapshots[idx] = str(path.resolve())
        records.append(int(np.count_nonzero(mask)))
    writer.release()

    summary = {
        "video": str(video_path),
        "seed_mask": str(seed_mask_path),
        "ckpt_path": str(ckpt_path),
        "model": args.model,
        "start_step": args.start_step,
        "source_frames": source_frames,
        "processed_frames": len(all_masks),
        "fps": fps,
        "size": list(size),
        "elapsed_s": elapsed,
        "pipeline_fps": len(all_masks) / max(elapsed, 1e-6),
        "mask_pixels_mean": float(np.mean(records)) if records else 0.0,
        "mask_pixels_max": int(np.max(records)) if records else 0,
        "log": log,
        "log_time": log_time,
        "overlay_video": str((out_dir / "overlay.mp4").resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "snapshots": snapshots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
