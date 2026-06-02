#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from shadowcam.dataset import list_pairs
from shadowcam.preprocess import image_to_float_input, load_grayscale


def representative_dataset(data_root: str, limit: int):
    pairs = list_pairs(data_root, "train")[:limit]
    for image_path, _ in pairs:
        gray = load_grayscale(image_path)
        x = image_to_float_input(gray)[None, ..., None].astype(np.float32)
        yield [x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--representative-samples", type=int, default=256)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(args.data, args.representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)
    print(f"Wrote {out} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()

