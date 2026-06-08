#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # pragma: no cover - TensorFlow fallback for local dev.
    import tensorflow.lite as tflite

from shadowcam.dataset import list_pairs
from shadowcam.metrics import segmentation_report
from shadowcam.preprocess import image_to_float_input, load_grayscale, load_mask


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        return x.astype(np.float32)
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = (np.arange(height) * (mask.shape[0] / height)).astype(np.int32)
    xs = (np.arange(width) * (mask.shape[1] / width)).astype(np.int32)
    return mask[ys[:, None], xs[None, :]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]

    y_true_all = []
    y_pred_all = []
    pairs = list_pairs(args.data, args.split)
    start = time.perf_counter()
    for image_path, mask_path in pairs:
        gray = load_grayscale(image_path)
        x = image_to_float_input(gray)[None, ..., None].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], quantize_input(x, input_scale, input_zero))
        interpreter.invoke()
        logits = interpreter.get_tensor(output_detail["index"])
        logits = dequantize_output(logits, output_scale, output_zero)
        pred = np.argmax(logits[0], axis=-1).astype(np.uint8)
        mask = load_mask(mask_path)
        if mask.shape != pred.shape:
            mask = resize_mask_nearest(mask, pred.shape[0], pred.shape[1])
        y_true_all.append(mask[None, ...].astype(np.uint8))
        y_pred_all.append(pred[None, ...])

    elapsed = time.perf_counter() - start
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    report = segmentation_report(y_true, y_pred)
    report["desktop_fps"] = len(pairs) / max(elapsed, 1e-6)
    report["input_quantization"] = {"scale": input_scale, "zero_point": input_zero}
    report["output_quantization"] = {"scale": output_scale, "zero_point": output_zero}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
