#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEFAULT = (
    "/Users/kongfans/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_zxjeq5ptpdkt22_d12e/temp/RWTemp/2026-06/"
    "9849b8961b4b8ab8a44aef4acacc7eba/c9be5e96759298bd43cce0bb156c07e9.mp4"
)
DAS_DIR = REPO_ROOT / "work/external/Detect-AnyShadow"
SAM_CKPT = DAS_DIR / "checkpoints/chk_sam/finetune.pth"
LSTN_CKPT = DAS_DIR / "checkpoints/lstnb"
OUT_ROOT = REPO_ROOT / "work/video_shadow_compare/anyshadow_fastapi_demo"


class _TimerEvent:
    def __init__(self, enable_timing: bool = True):
        self.t = 0.0

    def record(self) -> None:
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


app = FastAPI(title="AnyShadow Local Demo")
SAM_PREDICTOR: SamPredictor | None = None
RUNS: dict[str, dict[str, Any]] = {}


class LoadRequest(BaseModel):
    video_path: str = VIDEO_DEFAULT
    max_frames: int = 90
    stride: int = 1


class RefineRequest(BaseModel):
    run_id: str
    mask_png_base64: str


class PropagateRequest(BaseModel):
    run_id: str
    max_resolution: int = 180


def get_sam_predictor() -> SamPredictor:
    global SAM_PREDICTOR
    if SAM_PREDICTOR is None:
        if not SAM_CKPT.exists():
            raise RuntimeError(f"missing SAM checkpoint: {SAM_CKPT}")
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
        model.to(device=device)
        model.eval()
        SAM_PREDICTOR = SamPredictor(model)
    return SAM_PREDICTOR


def encode_png_b64(image_bgr_or_gray: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr_or_gray)
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_png_b64(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("failed to decode mask PNG")
    return image


def load_video_frames(video_path: Path, max_frames: int, stride: int) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        raise RuntimeError("no frames loaded")
    return frames, fps / max(1, stride), source_frames


def boxes_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape[:2]
    boxes: list[tuple[int, np.ndarray]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((
            area,
            np.asarray([max(0, x - 20), max(0, y - 20), min(w - 1, x + bw + 20), min(h - 1, y + bh + 20)], dtype=np.float32),
        ))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:8]]


def draw_overlay(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(frame_bgr)
    fill[mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(frame_bgr, 0.84, fill, 0.16, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def run_lstn(frames: list[np.ndarray], seed_mask: np.ndarray, fps: float, run_dir: Path, max_resolution: int) -> dict[str, Any]:
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
        evaluator = Evaluator(cfg, frames, [seed_mask])
        propagated, log, log_time = evaluator.evaluating()
    finally:
        os.chdir(old_cwd)
    elapsed = time.perf_counter() - start

    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    video_out = run_dir / "overlay.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    all_masks = [seed_mask] + propagated
    for idx, (frame_rgb, mask) in enumerate(zip(frames, all_masks)):
        cv2.imwrite(str(masks_dir / f"frame_{idx:05d}.png"), mask)
        writer.write(draw_overlay(frame_rgb, mask))
    writer.release()
    summary = {
        "frames": len(all_masks),
        "elapsed_s": elapsed,
        "pipeline_fps": len(all_masks) / max(elapsed, 1e-6),
        "overlay": str(video_out),
        "masks": str(masks_dir),
        "lstn_log": log,
        "lstn_time": log_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/load")
def api_load(req: LoadRequest) -> JSONResponse:
    try:
        video_path = Path(req.video_path).expanduser().resolve()
        frames, fps, source_frames = load_video_frames(video_path, req.max_frames, req.stride)
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        RUNS[run_id] = {
            "video_path": str(video_path),
            "frames": frames,
            "fps": fps,
            "run_dir": run_dir,
        }
        first_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        h, w = frames[0].shape[:2]
        return JSONResponse({
            "run_id": run_id,
            "frame_png_base64": encode_png_b64(first_bgr),
            "width": w,
            "height": h,
            "loaded_frames": len(frames),
            "source_frames": source_frames,
            "fps": fps,
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/refine")
def api_refine(req: RefineRequest) -> JSONResponse:
    try:
        run = RUNS[req.run_id]
        frames = run["frames"]
        user_mask = decode_png_b64(req.mask_png_base64)
        if user_mask.ndim == 3:
            if user_mask.shape[2] == 4:
                user_mask = user_mask[:, :, 3]
            else:
                user_mask = cv2.cvtColor(user_mask, cv2.COLOR_BGR2GRAY)
        user_mask = ((user_mask > 0).astype(np.uint8) * 255)
        if user_mask.shape[:2] != frames[0].shape[:2]:
            h, w = frames[0].shape[:2]
            user_mask = cv2.resize(user_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        boxes = boxes_from_mask(user_mask)
        if not boxes:
            raise RuntimeError("没有检测到涂抹区域，请在第一帧阴影上画一块")
        predictor = get_sam_predictor()
        predictor.set_image(frames[0])
        seed = np.zeros(user_mask.shape, dtype=np.uint8)
        for box in boxes:
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            seed[masks[int(np.argmax(scores))] > 0] = 255
        run["seed_mask"] = seed
        preview = draw_overlay(frames[0], seed)
        return JSONResponse({
            "seed_png_base64": encode_png_b64(preview),
            "seed_pixels": int(np.count_nonzero(seed)),
        })
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run_id not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/propagate")
def api_propagate(req: PropagateRequest) -> JSONResponse:
    try:
        run = RUNS[req.run_id]
        if "seed_mask" not in run:
            raise RuntimeError("请先生成 seed mask")
        summary = run_lstn(run["frames"], run["seed_mask"], run["fps"], run["run_dir"], req.max_resolution)
        return JSONResponse({
            "summary": summary,
            "video_url": f"/runs/{req.run_id}/overlay.mp4",
        })
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run_id not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id}/overlay.mp4")
def get_overlay(run_id: str) -> FileResponse:
    path = OUT_ROOT / run_id / "overlay.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="overlay not found")
    return FileResponse(path, media_type="video/mp4")


HTML = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>AnyShadow Local Demo</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111; color: #eee; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 18px; }}
    .row {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
    input, button {{ font: inherit; padding: 8px 10px; border-radius: 6px; border: 1px solid #555; background: #1d1d1d; color: #eee; }}
    button {{ cursor: pointer; background: #2b5cff; border-color: #2b5cff; }}
    button.secondary {{ background: #333; border-color: #555; }}
    canvas {{ max-width: 100%; border: 1px solid #444; background: #222; touch-action: none; }}
    .panel {{ flex: 1 1 560px; }}
    label {{ display:block; margin: 8px 0 4px; color:#bbb; }}
    #videoPath {{ width: min(900px, 100%); }}
    pre {{ white-space: pre-wrap; background:#181818; padding:12px; border-radius:8px; color:#b8ffb8; }}
    video {{ width: 100%; max-height: 560px; background:#000; }}
  </style>
</head>
<body>
<main>
  <h2>AnyShadow 本地 Demo：第一帧涂阴影，然后传播到后续帧</h2>
  <label>视频路径</label>
  <input id="videoPath" value="{VIDEO_DEFAULT}" />
  <label>最大帧数和步长</label>
  <input id="maxFrames" type="number" value="90" min="0" style="width:100px" />
  <input id="stride" type="number" value="1" min="1" style="width:80px" />
  <button onclick="loadVideo()">1. 载入视频第一帧</button>
  <button class="secondary" onclick="clearMask()">清空涂抹</button>
  <button onclick="refineSeed()">2. SAM 细化第一帧 mask</button>
  <button onclick="propagate()">3. LSTN 传播</button>
  <div class="row" style="margin-top:16px">
    <div class="panel">
      <h3>第一帧：用鼠标涂黄色阴影区域</h3>
      <canvas id="canvas"></canvas>
    </div>
    <div class="panel">
      <h3>Seed / 输出</h3>
      <img id="seedPreview" style="max-width:100%; border:1px solid #444; display:none" />
      <video id="outVideo" controls style="display:none"></video>
    </div>
  </div>
  <pre id="log">等待载入。</pre>
</main>
<script>
let runId = null;
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let baseImg = new Image();
let drawing = false;

function log(x) {{ document.getElementById('log').textContent = typeof x === 'string' ? x : JSON.stringify(x, null, 2); }}

function eventPos(e) {{
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width;
  const sy = canvas.height / r.height;
  return {{x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy}};
}}

function drawAt(e) {{
  if (!drawing) return;
  const p = eventPos(e);
  ctx.save();
  ctx.globalAlpha = 0.7;
  ctx.fillStyle = 'yellow';
  ctx.beginPath();
  ctx.arc(p.x, p.y, 28, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}}

canvas.addEventListener('pointerdown', e => {{ drawing = true; canvas.setPointerCapture(e.pointerId); drawAt(e); }});
canvas.addEventListener('pointermove', drawAt);
canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointercancel', () => drawing = false);

async function loadVideo() {{
  log('正在载入视频...');
  const res = await fetch('/api/load', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      video_path: document.getElementById('videoPath').value,
      max_frames: Number(document.getElementById('maxFrames').value),
      stride: Number(document.getElementById('stride').value)
    }})
  }});
  const data = await res.json();
  if (!res.ok) {{ log(data); return; }}
  runId = data.run_id;
  baseImg.onload = () => {{
    canvas.width = data.width;
    canvas.height = data.height;
    ctx.drawImage(baseImg, 0, 0);
  }};
  baseImg.src = 'data:image/png;base64,' + data.frame_png_base64;
  log(data);
}}

function clearMask() {{
  if (baseImg.src) ctx.drawImage(baseImg, 0, 0);
}}

function makeMaskPng() {{
  const w = canvas.width, h = canvas.height;
  const current = ctx.getImageData(0, 0, w, h);
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');
  const out = tctx.createImageData(w, h);
  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = w; baseCanvas.height = h;
  const bctx = baseCanvas.getContext('2d');
  bctx.drawImage(baseImg, 0, 0);
  const base = bctx.getImageData(0, 0, w, h);
  for (let i = 0; i < current.data.length; i += 4) {{
    const dr = Math.abs(current.data[i] - base.data[i]);
    const dg = Math.abs(current.data[i+1] - base.data[i+1]);
    const db = Math.abs(current.data[i+2] - base.data[i+2]);
    const on = (dr + dg + db) > 60;
    out.data[i] = 255; out.data[i+1] = 255; out.data[i+2] = 255; out.data[i+3] = on ? 255 : 0;
  }}
  tctx.putImageData(out, 0, 0);
  return tmp.toDataURL('image/png');
}}

async function refineSeed() {{
  if (!runId) {{ log('先载入视频'); return; }}
  log('正在加载 SAM 并细化第一帧，第一次可能较慢...');
  const res = await fetch('/api/refine', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{ run_id: runId, mask_png_base64: makeMaskPng() }})
  }});
  const data = await res.json();
  if (!res.ok) {{ log(data); return; }}
  const img = document.getElementById('seedPreview');
  img.src = 'data:image/png;base64,' + data.seed_png_base64;
  img.style.display = 'block';
  log(data);
}}

async function propagate() {{
  if (!runId) {{ log('先载入视频'); return; }}
  log('正在运行 LSTN 传播，CPU fallback 会比较慢...');
  const res = await fetch('/api/propagate', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{ run_id: runId, max_resolution: 180 }})
  }});
  const data = await res.json();
  if (!res.ok) {{ log(data); return; }}
  const video = document.getElementById('outVideo');
  video.src = data.video_url + '?t=' + Date.now();
  video.style.display = 'block';
  log(data);
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=7871, log_level="info")
