from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from .config import INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH


PathLike = Union[str, Path]


def load_grayscale(path: PathLike) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def load_mask(path: PathLike) -> np.ndarray:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.max() > 2:
        arr = (arr > 127).astype(np.uint8)
    return arr


def resize_image_96(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)


def resize_mask_48(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)


def local_normalize(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    blur = cv2.blur(gray_f, (9, 9))
    norm = gray_f - blur + 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def maybe_jpeg_roundtrip(gray: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return gray
    image = Image.fromarray(gray, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


def random_augment(gray: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        gray = np.fliplr(gray)
        mask = np.fliplr(mask)

    if rng.random() < 0.8:
        gamma = rng.uniform(0.65, 1.55)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)

    if rng.random() < 0.8:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.uniform(-32, 32)
        gray = np.clip(gray.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if rng.random() < 0.35:
        noise = rng.normal(0, rng.uniform(2, 12), gray.shape)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        gray = maybe_jpeg_roundtrip(gray, int(rng.integers(35, 85)))

    if rng.random() < 0.5:
        gray = local_normalize(gray)

    return gray, mask


def image_to_float_input(gray: np.ndarray) -> np.ndarray:
    gray = resize_image_96(gray)
    return (gray.astype(np.float32) / 127.5) - 1.0
