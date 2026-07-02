#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEFAULT = (
    "/Users/kongfans/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_zxjeq5ptpdkt22_d12e/temp/RWTemp/2026-06/"
    "9849b8961b4b8ab8a44aef4acacc7eba/c9be5e96759298bd43cce0bb156c07e9.mp4"
)
DAS_DIR = REPO_ROOT / "work/external/Detect-AnyShadow"
SAM_CKPT = DAS_DIR / "checkpoints/chk_sam/finetune.pth"
LSTN_CKPT = DAS_DIR / "checkpoints/lstnb"
OUT_DIR = REPO_ROOT / "work/video_shadow_compare/detect_anyshadow_demo_local"


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


patch_cuda_to_cpu()
sys.path.insert(0, str((DAS_DIR / "lstn").resolve()))
sys.path.insert(0, str(DAS_DIR.resolve()))

from sam import SamPredictor, sam_model_registry  # type: ignore  # noqa: E402
from lstn.configs.visha import EngineConfig  # type: ignore  # noqa: E402
from lstn.networks.managers.eval_demo import Evaluator  # type: ignore  # noqa: E402


SAM_PREDICTOR: SamPredictor | None = None


def get_sam_predictor() -> SamPredictor:
    global SAM_PREDICTOR
    if SAM_PREDICTOR is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
        model.to(device=device)
        model.eval()
        SAM_PREDICTOR = SamPredictor(model)
    return SAM_PREDICTOR


def read_video_frames(video_path: str, max_frames: int, stride: int) -> tuple[list[np.ndarray], float, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"打不开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    read_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % max(1, stride) == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        read_index += 1
    cap.release()
    if not frames:
        raise gr.Error("没有读到视频帧")
    info = f"frames loaded: {len(frames)} / source {total}, fps: {fps:.2f}, size: {frames[0].shape[1]}x{frames[0].shape[0]}"
    return frames, fps / max(1, stride), info


def load_video(video_path: str, max_frames: int, stride: int):
    frames, fps, info = read_video_frames(video_path, int(max_frames), int(stride))
    state = {"video_path": video_path, "frames": frames, "fps": fps}
    return state, frames[0], info


def extract_editor_image_and_mask(editor_value: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(editor_value, dict):
        image = editor_value.get("background") or editor_value.get("composite")
        layers = editor_value.get("layers") or []
        if image is None:
            raise gr.Error("没有拿到第一帧图像")
        image_np = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)[:, :, :3]
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for layer in layers:
            layer_np = np.asarray(layer)
            if layer_np.ndim == 3 and layer_np.shape[2] == 4:
                mask[layer_np[:, :, 3] > 0] = 255
            elif layer_np.ndim == 2:
                mask[layer_np > 0] = 255
        if np.count_nonzero(mask) == 0 and editor_value.get("composite") is not None:
            comp = np.asarray(editor_value["composite"])
            diff = np.abs(comp[:, :, :3].astype(np.int16) - image_np.astype(np.int16)).sum(axis=2)
            mask[diff > 10] = 255
        return image_np, mask
    raise gr.Error("请先在第一帧上涂一块阴影区域")


def boxes_from_mask(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes = []
    h, w = mask.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append([max(0, x - 20), max(0, y - 20), min(w - 1, x + bw + 20), min(h - 1, y + bh + 20)])
    if not boxes:
        raise gr.Error("没有检测到你涂的区域，请用画笔涂第一帧里的阴影")
    return np.asarray(boxes, dtype=np.float32)


def refine_first_mask(editor_value: Any, state: dict):
    image_rgb, user_mask = extract_editor_image_and_mask(editor_value)
    predictor = get_sam_predictor()
    predictor.set_image(image_rgb)
    merged = np.zeros(user_mask.shape, dtype=np.uint8)
    for box in boxes_from_mask(user_mask):
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        merged[masks[int(np.argmax(scores))] > 0] = 255
    state = dict(state or {})
    state["seed_mask"] = merged
    preview = draw_overlay(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), merged)
    return state, cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), f"seed mask pixels: {int(np.count_nonzero(merged))}"


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(frame_bgr, 0.84, fill, 0.16, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def run_lstn(state: dict, max_resolution: int):
    if not state or "frames" not in state or "seed_mask" not in state:
        raise gr.Error("先加载视频，再涂第一帧并生成 seed mask")
    frames: list[np.ndarray] = state["frames"]
    seed = state["seed_mask"]
    fps = float(state["fps"])
    cfg = EngineConfig("lstn", "lstnb")
    cfg.TEST_EMA = False
    cfg.TEST_CKPT_STEP = 10000
    cfg.TEST_CKPT_PATH = str(LSTN_CKPT.resolve())
    cfg.TEST_DATASET_PATH = ""
    cfg.TEST_MAX_LONG_EDGE = float(max_resolution) * 800.0 / 480.0
    cfg.TEST_WORKERS = 0
    cfg.TEST_FRAME_LOG = False
    cfg.MODEL_LSAB_ENABLE_CORR = False

    old_cwd = Path.cwd()
    os.chdir(DAS_DIR)
    start = time.perf_counter()
    try:
        evaluator = Evaluator(cfg, frames, [seed])
        propagated, log, log_time = evaluator.evaluating()
    finally:
        os.chdir(old_cwd)
    elapsed = time.perf_counter() - start

    run_dir = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    masks_dir = run_dir / "masks"
    run_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    video_out = run_dir / "overlay.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    all_masks = [seed] + propagated
    for idx, (frame_rgb, mask) in enumerate(zip(frames, all_masks)):
        cv2.imwrite(str(masks_dir / f"frame_{idx:05d}.png"), mask)
        overlay = draw_overlay(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), mask)
        writer.write(overlay)
    writer.release()
    summary = {
        "frames": len(all_masks),
        "elapsed_s": elapsed,
        "fps": len(all_masks) / max(elapsed, 1e-6),
        "video": str(video_out),
        "masks": str(masks_dir),
        "lstn_log": log,
        "lstn_time": log_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(video_out), json.dumps(summary, indent=2, ensure_ascii=False)


with gr.Blocks(title="Detect-AnyShadow Local Demo") as demo:
    gr.Markdown("# Detect-AnyShadow Local Demo")
    state = gr.State({})
    with gr.Row():
        video_path = gr.Textbox(value=VIDEO_DEFAULT, label="Video path")
        max_frames = gr.Number(value=90, precision=0, label="Max frames, 0 = all")
        stride = gr.Number(value=1, precision=0, label="Frame stride")
    load_btn = gr.Button("1. Load video")
    info = gr.Textbox(label="Info")
    first_frame = gr.ImageEditor(label="2. Paint shadow on first frame", type="pil", brush=gr.Brush(colors=["#ffff00"], default_size=40))
    seed_btn = gr.Button("3. Refine first mask with fine-tuned SAM")
    seed_preview = gr.Image(label="Seed preview")
    seed_info = gr.Textbox(label="Seed info")
    max_resolution = gr.Number(value=180, precision=0, label="LSTN max resolution")
    run_btn = gr.Button("4. Run LSTN video propagation")
    output_video = gr.Video(label="Output overlay")
    output_log = gr.Code(label="Run summary", language="json")

    load_btn.click(load_video, inputs=[video_path, max_frames, stride], outputs=[state, first_frame, info])
    seed_btn.click(refine_first_mask, inputs=[first_frame, state], outputs=[state, seed_preview, seed_info])
    run_btn.click(run_lstn, inputs=[state, max_resolution], outputs=[output_video, output_log])


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    import gradio.networking as gradio_networking

    gradio_networking.url_ok = lambda url: True
    demo.get_api_info = lambda: {}
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        show_error=True,
        show_api=False,
        quiet=False,
    )
