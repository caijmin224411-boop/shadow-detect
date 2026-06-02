from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from .config import INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH
from .preprocess import (
    image_to_float_input,
    load_grayscale,
    load_mask,
    random_augment,
    resize_mask_48,
)


PathLike = Union[str, Path]


def list_pairs(root: PathLike, split: str):
    root = Path(root)
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {mask_dir}")

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {root / split}")
    return pairs


def _load_np(image_path_b, mask_path_b, augment: bool, seed: int):
    image_path = image_path_b.decode("utf-8")
    mask_path = mask_path_b.decode("utf-8")
    gray = load_grayscale(image_path)
    mask = load_mask(mask_path)
    if augment:
        rng = np.random.default_rng(seed)
        gray, mask = random_augment(gray, mask, rng)
    x = image_to_float_input(gray)[..., None]
    y = resize_mask_48(mask).astype(np.int32)
    return x.astype(np.float32), y


def make_dataset(
    root: PathLike,
    split: str,
    batch_size: int,
    augment: bool,
    shuffle: bool = True,
):
    pairs = list_pairs(root, split)
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(min(len(pairs), 4096), reshuffle_each_iteration=True)

    counter = tf.data.Dataset.counter()
    ds = tf.data.Dataset.zip((ds, counter))

    def mapper(pair, idx):
        image_path, mask_path = pair
        x, y = tf.numpy_function(
            _load_np,
            [image_path, mask_path, augment, tf.cast(idx, tf.int64)],
            [tf.float32, tf.int32],
        )
        x.set_shape((INPUT_HEIGHT, INPUT_WIDTH, 1))
        y.set_shape((OUTPUT_HEIGHT, OUTPUT_WIDTH))
        return x, y

    return ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_modular_dataset(
    root: PathLike,
    split: str,
    batch_size: int,
    augment: bool,
    shuffle: bool = True,
):
    def to_heads(x, y):
        shadow = tf.cast(y == 1, tf.float32)[..., None]
        exclude = tf.cast(y == 2, tf.float32)[..., None]
        return x, {"shadow_logits": shadow, "exclude_logits": exclude}

    return make_dataset(root, split, batch_size, augment, shuffle).map(
        to_heads,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
