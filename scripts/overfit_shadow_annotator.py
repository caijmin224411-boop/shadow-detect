#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from PIL import Image


LABELS = {
    0: "erase/clean",
    1: "paper shadow",
    2: "hand/object exclude",
    3: "non-paper optional",
}


class SaveMaskRequest(BaseModel):
    mask_png_base64: str


def encode_png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def decode_data_url_png(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    data = base64.b64decode(data_url)
    return Image.open(io.BytesIO(data))


def pick_indices(frame_count: int, frames: int, stride: int) -> list[int]:
    if frame_count <= 0:
        return []
    if stride > 0:
        return list(range(0, frame_count, stride))[:frames]
    if frame_count <= frames:
        return list(range(frame_count))
    values = np.linspace(0, frame_count - 1, frames)
    return sorted({int(round(v)) for v in values})


def find_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def read_manifest(review_dir: Path) -> list[dict[str, Any]]:
    manifest = review_dir / "review_manifest.csv"
    rows: list[dict[str, Any]] = []
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = row.get("stem") or row.get("name")
                if stem:
                    row["stem"] = stem
                    rows.append(row)
    if rows:
        return rows

    image_dir = review_dir / "images"
    if not image_dir.exists():
        return []
    for idx, path in enumerate(sorted(image_dir.iterdir())):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            rows.append({"stem": path.stem, "source_frame": idx})
    return rows


def write_manifest(review_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest = review_dir / "review_manifest.csv"
    fieldnames = ["stem", "source_frame", "image", "reviewed_mask", "review_status"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def ensure_review_dirs(review_dir: Path) -> None:
    for name in ("images", "masks_reviewed", "previews"):
        (review_dir / name).mkdir(parents=True, exist_ok=True)


def remove_yellow_overlay(frame: np.ndarray) -> np.ndarray:
    """Remove retained yellow comparison contours before manual labeling."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray((20, 145, 145)), np.asarray((42, 255, 255)))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if not mask.any():
        return frame
    return cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA)


def extract_video_frames(
    video: Path,
    review_dir: Path,
    frames: int,
    stride: int,
    overwrite: bool,
    clean_yellow_overlay: bool,
) -> list[dict[str, Any]]:
    ensure_review_dirs(review_dir)
    existing = read_manifest(review_dir)
    if existing and not overwrite:
        return existing

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(pick_indices(frame_count, frames, stride))
    rows: list[dict[str, Any]] = []
    kept = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in indices:
            frame_idx += 1
            continue

        if clean_yellow_overlay:
            frame = remove_yellow_overlay(frame)
        stem = f"overfit_{kept:04d}_f{frame_idx:06d}"
        image_path = review_dir / "images" / f"{stem}.png"
        mask_path = review_dir / "masks_reviewed" / f"{stem}.png"
        cv2.imwrite(str(image_path), frame)
        blank = np.zeros(frame.shape[:2], dtype=np.uint8)
        Image.fromarray(blank).save(mask_path)
        rows.append(
            {
                "stem": stem,
                "source_frame": frame_idx,
                "image": str(image_path),
                "reviewed_mask": str(mask_path),
                "review_status": "blank",
            }
        )
        kept += 1
        frame_idx += 1

    cap.release()
    if not rows:
        raise RuntimeError(f"no frames extracted from {video}")

    write_manifest(review_dir, rows)
    (review_dir / "README.md").write_text(
        "Manual overfit shadow labels.\n"
        "Mask labels: 0=clean/background, 1=paper_shadow, 2=hand/object/arm_exclude, 3=non_paper_optional.\n"
        "Convert with scripts/real_video_review_to_unified.py after review.\n",
        encoding="utf-8",
    )
    return rows


def load_mask(mask_path: Path, image_size: tuple[int, int]) -> Image.Image:
    width, height = image_size
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.uint8)
        arr = np.where(arr > 3, 1, arr).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")


def write_preview(image_path: Path, mask_path: Path, preview_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    colors = {
        1: np.array([0, 255, 255], dtype=np.uint8),
        2: np.array([0, 90, 255], dtype=np.uint8),
        3: np.array([255, 0, 255], dtype=np.uint8),
    }
    for label, color in colors.items():
        region = mask == label
        overlay[region] = (overlay[region].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5).astype(np.uint8)
    contours, _ = cv2.findContours((mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    contours, _ = cv2.findContours((mask == 2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 90, 255), 2)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), overlay)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Shadow Overfit Annotator</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #15171b; color: #e8eaed; }
    header { height: 52px; display: flex; align-items: center; gap: 14px; padding: 0 16px; background: #202329; border-bottom: 1px solid #343842; }
    button, select, input { background: #2b3038; color: #f4f6f8; border: 1px solid #4a5260; border-radius: 6px; padding: 7px 10px; font-size: 14px; }
    button:hover { background: #38404b; }
    button.active { outline: 2px solid #ffd400; }
    .main { display: grid; grid-template-columns: 240px 1fr; height: calc(100vh - 52px); }
    .side { overflow: auto; border-right: 1px solid #343842; background: #191c21; padding: 10px; }
    .frameBtn { width: 100%; text-align: left; margin-bottom: 6px; }
    .frameBtn.done { border-color: #40c463; }
    .stageWrap { overflow: auto; padding: 14px; }
    .stage { position: relative; display: inline-block; background: #090a0c; box-shadow: 0 0 0 1px #343842; }
    canvas { position: absolute; left: 0; top: 0; image-rendering: auto; }
    #imageCanvas { position: relative; }
    .hint { color: #aab1bd; font-size: 13px; }
    .spacer { flex: 1; }
    .danger { color: #ffb4a8; }
  </style>
</head>
<body>
  <header>
    <strong>Shadow Overfit Annotator</strong>
    <button id="label1" onclick="setLabel(1)">1 阴影</button>
    <button id="label2" onclick="setLabel(2)">2 手/物体排除</button>
    <button id="label3" onclick="setLabel(3)">3 非纸面</button>
    <button id="label0" onclick="setLabel(0)">0 橡皮</button>
    <label>笔刷 <input id="brush" type="range" min="2" max="120" value="28" /></label>
    <button onclick="undoStroke()">撤销</button>
    <button onclick="copyPreviousMask()">复制上一帧</button>
    <button onclick="clearMask()">清空</button>
    <button onclick="saveMask()">保存</button>
    <button onclick="prevFrame()">上一帧</button>
    <button onclick="nextFrame()">下一帧</button>
    <span class="spacer"></span>
    <span id="status" class="hint">loading...</span>
  </header>
  <div class="main">
    <aside class="side">
      <div class="hint">
        快捷键：1 阴影，2 排除，0 擦除，[/] 调笔刷，⌘/Ctrl+Z 撤销，C 复制上一帧，S 保存，A/D 或 ←/→ 切帧。<br>
        黄色=要过拟合的纸面阴影；橙色=手/机械臂/笔等不要识别成阴影。
      </div>
      <hr />
      <div id="frames"></div>
    </aside>
    <section class="stageWrap">
      <div id="stage" class="stage">
        <canvas id="imageCanvas"></canvas>
        <canvas id="overlayCanvas"></canvas>
      </div>
    </section>
  </div>
<script>
let meta = null;
let index = 0;
let label = 1;
let labels = null;
let width = 0;
let height = 0;
let drawing = false;
let dirty = false;
let undoStack = [];

const imageCanvas = document.getElementById("imageCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const imageCtx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

function setStatus(text, danger=false) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = danger ? "hint danger" : "hint";
}

function setLabel(v) {
  label = v;
  for (const id of ["label0", "label1", "label2", "label3"]) document.getElementById(id).classList.remove("active");
  document.getElementById("label" + v).classList.add("active");
}

function renderFrameButtons() {
  const box = document.getElementById("frames");
  box.innerHTML = "";
  meta.frames.forEach((f, i) => {
    const b = document.createElement("button");
    b.className = "frameBtn" + (f.review_status === "saved" ? " done" : "");
    b.textContent = `${i + 1}. ${f.stem}`;
    b.onclick = async () => { await maybeSave(); await loadFrame(i); };
    box.appendChild(b);
  });
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error(await res.text());
  meta = await res.json();
  renderFrameButtons();
  await loadFrame(0);
  setLabel(1);
}

async function loadFrame(i) {
  index = Math.max(0, Math.min(meta.frames.length - 1, i));
  const res = await fetch(`/api/frame/${index}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  const img = new Image();
  img.onload = () => {
    width = data.width;
    height = data.height;
    imageCanvas.width = overlayCanvas.width = width;
    imageCanvas.height = overlayCanvas.height = height;
    document.getElementById("stage").style.width = width + "px";
    document.getElementById("stage").style.height = height + "px";
    imageCtx.drawImage(img, 0, 0);
    loadMaskImage(data.mask);
    dirty = false;
    undoStack = [];
    setStatus(`${index + 1}/${meta.frames.length} ${data.stem}`);
  };
  img.src = data.image;
}

function loadMaskImage(maskUrl) {
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = width; c.height = height;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    labels = new Uint8Array(width * height);
    for (let p = 0, j = 0; p < pixels.length; p += 4, j++) labels[j] = Math.min(3, pixels[p]);
    drawOverlay();
  };
  img.src = maskUrl;
}

function drawOverlay() {
  const img = overlayCtx.createImageData(width, height);
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    const p = i * 4;
    if (v === 1) { img.data[p] = 255; img.data[p+1] = 220; img.data[p+2] = 0; img.data[p+3] = 118; }
    else if (v === 2) { img.data[p] = 255; img.data[p+1] = 90; img.data[p+2] = 0; img.data[p+3] = 118; }
    else if (v === 3) { img.data[p] = 230; img.data[p+1] = 70; img.data[p+2] = 255; img.data[p+3] = 108; }
    else { img.data[p+3] = 0; }
  }
  overlayCtx.putImageData(img, 0, 0);
}

function canvasPoint(evt) {
  const rect = overlayCanvas.getBoundingClientRect();
  return {
    x: Math.floor((evt.clientX - rect.left) * width / rect.width),
    y: Math.floor((evt.clientY - rect.top) * height / rect.height),
  };
}

function paint(evt) {
  if (!labels) return;
  const pt = canvasPoint(evt);
  const r = parseInt(document.getElementById("brush").value, 10);
  const r2 = r * r;
  const x0 = Math.max(0, pt.x - r), x1 = Math.min(width - 1, pt.x + r);
  const y0 = Math.max(0, pt.y - r), y1 = Math.min(height - 1, pt.y + r);
  for (let y = y0; y <= y1; y++) {
    const dy = y - pt.y;
    for (let x = x0; x <= x1; x++) {
      const dx = x - pt.x;
      if (dx * dx + dy * dy <= r2) labels[y * width + x] = label;
    }
  }
  dirty = true;
  drawOverlay();
}

function pushUndo() {
  if (!labels) return;
  undoStack.push(labels.slice());
  if (undoStack.length > 30) undoStack.shift();
}

function undoStroke() {
  if (!undoStack.length) return;
  labels = undoStack.pop();
  dirty = true;
  drawOverlay();
  setStatus("已撤销，可继续修改");
}

function clearMask() {
  if (!labels) return;
  pushUndo();
  labels.fill(0);
  dirty = true;
  drawOverlay();
  setStatus("当前掩膜已清空，尚未保存");
}

async function copyPreviousMask() {
  if (index <= 0) {
    setStatus("当前已经是第一帧", true);
    return;
  }
  const res = await fetch(`/api/frame/${index - 1}`);
  if (!res.ok) {
    setStatus(await res.text(), true);
    return;
  }
  const data = await res.json();
  const img = new Image();
  img.onload = () => {
    pushUndo();
    const c = document.createElement("canvas");
    c.width = width; c.height = height;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, width, height);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    labels = new Uint8Array(width * height);
    for (let p = 0, j = 0; p < pixels.length; p += 4, j++) labels[j] = Math.min(3, pixels[p]);
    dirty = true;
    drawOverlay();
    setStatus(`已复制第 ${index} 帧掩膜，请修正后保存`);
  };
  img.src = data.mask;
}

overlayCanvas.addEventListener("pointerdown", e => {
  pushUndo();
  drawing = true;
  overlayCanvas.setPointerCapture(e.pointerId);
  paint(e);
});
overlayCanvas.addEventListener("pointermove", e => { if (drawing) paint(e); });
overlayCanvas.addEventListener("pointerup", () => { drawing = false; });
overlayCanvas.addEventListener("pointercancel", () => { drawing = false; });

function maskDataUrl() {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < labels.length; i++) {
    const p = i * 4;
    const v = labels[i];
    img.data[p] = img.data[p+1] = img.data[p+2] = v;
    img.data[p+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

async function saveMask() {
  if (!labels) return;
  setStatus("saving...");
  const res = await fetch(`/api/frame/${index}/mask`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({mask_png_base64: maskDataUrl()}),
  });
  if (!res.ok) {
    setStatus(await res.text(), true);
    return;
  }
  dirty = false;
  meta.frames[index].review_status = "saved";
  renderFrameButtons();
  setStatus(`saved ${index + 1}/${meta.frames.length}`);
}

async function maybeSave() {
  if (dirty) await saveMask();
}

async function nextFrame() { await maybeSave(); await loadFrame(index + 1); }
async function prevFrame() { await maybeSave(); await loadFrame(index - 1); }

document.addEventListener("keydown", async e => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "1") setLabel(1);
  else if (e.key === "2") setLabel(2);
  else if (e.key === "3") setLabel(3);
  else if (e.key === "0") setLabel(0);
  else if (e.key === "s" || e.key === "S") await saveMask();
  else if ((e.key === "z" || e.key === "Z") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undoStroke(); }
  else if (e.key === "c" || e.key === "C") await copyPreviousMask();
  else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") await nextFrame();
  else if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") await prevFrame();
  else if (e.key === "[" || e.key === "-") {
    const b = document.getElementById("brush");
    b.value = Math.max(parseInt(b.min), parseInt(b.value) - 4);
  } else if (e.key === "]" || e.key === "=") {
    const b = document.getElementById("brush");
    b.value = Math.min(parseInt(b.max), parseInt(b.value) + 4);
  }
});

loadMeta().catch(err => setStatus(String(err), true));
</script>
</body>
</html>
"""


def create_app(review_dir: Path) -> FastAPI:
    app = FastAPI(title="Shadow Overfit Annotator")

    def rows() -> list[dict[str, Any]]:
        data = read_manifest(review_dir)
        if not data:
            raise HTTPException(404, f"no frames found under {review_dir}")
        return data

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        data = rows()
        return {
            "review_dir": str(review_dir),
            "labels": LABELS,
            "frames": [
                {
                    "stem": row["stem"],
                    "source_frame": row.get("source_frame", ""),
                    "review_status": row.get("review_status", ""),
                }
                for row in data
            ],
        }

    @app.get("/api/frame/{idx}")
    def frame(idx: int) -> dict[str, Any]:
        data = rows()
        if idx < 0 or idx >= len(data):
            raise HTTPException(404, "frame index out of range")
        stem = data[idx]["stem"]
        image_path = find_image_path(review_dir / "images", stem)
        if image_path is None:
            raise HTTPException(404, f"missing image for {stem}")
        image = Image.open(image_path).convert("RGB")
        mask_path = review_dir / "masks_reviewed" / f"{stem}.png"
        mask = load_mask(mask_path, image.size)
        return {
            "stem": stem,
            "width": image.width,
            "height": image.height,
            "image": encode_png_data_url(image),
            "mask": encode_png_data_url(mask),
        }

    @app.post("/api/frame/{idx}/mask")
    def save_frame_mask(idx: int, request: SaveMaskRequest) -> dict[str, Any]:
        data = rows()
        if idx < 0 or idx >= len(data):
            raise HTTPException(404, "frame index out of range")
        stem = data[idx]["stem"]
        image_path = find_image_path(review_dir / "images", stem)
        if image_path is None:
            raise HTTPException(404, f"missing image for {stem}")
        image = Image.open(image_path).convert("RGB")
        mask = decode_data_url_png(request.mask_png_base64).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.uint8)
        arr = np.where(arr > 3, 1, arr).astype(np.uint8)
        mask_path = review_dir / "masks_reviewed" / f"{stem}.png"
        Image.fromarray(arr, mode="L").save(mask_path)
        preview_path = review_dir / "previews" / f"{stem}.jpg"
        write_preview(image_path, mask_path, preview_path)

        data[idx]["review_status"] = "saved"
        write_manifest(review_dir, data)
        return {"ok": True, "stem": stem, "mask": str(mask_path), "preview": str(preview_path)}

    @app.get("/previews/{name}")
    def preview(name: str) -> Response:
        path = review_dir / "previews" / name
        if not path.exists():
            raise HTTPException(404, "preview not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(path.read_bytes(), media_type=media_type)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser-based shadow mask annotator for single-scene overfitting.")
    parser.add_argument("--video", default="", help="Video path. Required when the review folder has no extracted frames.")
    parser.add_argument("--review", default="work/datasets/overfit_shadow_v1_review")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to sample when extracting a video.")
    parser.add_argument("--stride", type=int, default=0, help="Use every Nth frame instead of uniform sampling. 0 means uniform.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames in --review.")
    parser.add_argument("--remove-yellow-overlay", action="store_true", help="Inpaint retained yellow proxy contours during extraction.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7872)
    args = parser.parse_args()

    review_dir = Path(args.review)
    ensure_review_dirs(review_dir)
    existing = read_manifest(review_dir)
    if not existing:
        if not args.video:
            raise SystemExit("no review frames found; pass --video to extract frames first")
        print(f"Extracting frames from {args.video} -> {review_dir}")
        extract_video_frames(
            Path(args.video), review_dir, args.frames, args.stride, args.overwrite, args.remove_yellow_overlay
        )
    elif args.overwrite and args.video:
        print(f"Overwriting review frames from {args.video} -> {review_dir}")
        extract_video_frames(
            Path(args.video), review_dir, args.frames, args.stride, args.overwrite, args.remove_yellow_overlay
        )

    print(f"Review folder: {review_dir}")
    print(f"Open: http://{args.host}:{args.port}")
    app = create_app(review_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
