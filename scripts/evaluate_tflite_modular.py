#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from shadowcam.dataset import make_dataset
from shadowcam.metrics import segmentation_report


def quantize_input(x: np.ndarray, input_detail) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        return x.astype(input_detail["dtype"])
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(input_detail["dtype"])


def dequantize_output(y: np.ndarray, output_detail) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def classify_outputs(output_details):
    shadow_idx = None
    exclude_idx = None
    for i, detail in enumerate(output_details):
        name = detail["name"].lower()
        if "exclude" in name:
            exclude_idx = i
        elif "shadow" in name:
            shadow_idx = i
    if shadow_idx is None or exclude_idx is None:
        if len(output_details) != 2:
            raise RuntimeError(f"Expected two TFLite outputs, got {len(output_details)}")
        shadow_idx = 0 if shadow_idx is None else shadow_idx
        exclude_idx = 1 if exclude_idx is None else exclude_idx
    return shadow_idx, exclude_idx


def resolve_outputs_by_keras(interpreter, input_detail, output_details, keras_model, ds, max_samples: int):
    assignment_errors = np.zeros(2, dtype=np.float64)
    count = 0
    for x, _ in ds.take(max_samples):
        keras_shadow, keras_exclude = keras_model(x, training=False)
        interpreter.set_tensor(input_detail["index"], quantize_input(x.numpy(), input_detail))
        interpreter.invoke()
        outputs = [dequantize_output(interpreter.get_tensor(detail["index"]), detail) for detail in output_details]
        assignment_errors[0] += float(np.mean((outputs[0] - keras_shadow.numpy()) ** 2))
        assignment_errors[0] += float(np.mean((outputs[1] - keras_exclude.numpy()) ** 2))
        assignment_errors[1] += float(np.mean((outputs[1] - keras_shadow.numpy()) ** 2))
        assignment_errors[1] += float(np.mean((outputs[0] - keras_exclude.numpy()) ** 2))
        count += 1
    if count == 0:
        raise RuntimeError("No samples available to resolve TFLite output order")
    if assignment_errors[0] <= assignment_errors[1]:
        return 0, 1, assignment_errors.tolist()
    return 1, 0, assignment_errors.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--exclude-threshold", type=float, default=0.50)
    parser.add_argument("--keras-model", default=None, help="Optional Keras model used to resolve ambiguous TFLite output order.")
    parser.add_argument("--order-resolve-samples", type=int, default=16)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    shadow_idx, exclude_idx = classify_outputs(output_details)

    ds = make_dataset(args.data, args.split, batch_size=1, augment=False, shuffle=False)
    assignment_errors = None
    if args.keras_model:
        keras_model = tf.keras.models.load_model(args.keras_model, compile=False)
        shadow_idx, exclude_idx, assignment_errors = resolve_outputs_by_keras(
            interpreter,
            input_detail,
            output_details,
            keras_model,
            ds,
            args.order_resolve_samples,
        )
    y_true_all = []
    y_pred_all = []
    frame_count = 0
    start = time.perf_counter()
    for x, y in ds:
        interpreter.set_tensor(input_detail["index"], quantize_input(x.numpy(), input_detail))
        interpreter.invoke()
        raw_outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]
        shadow_logits = dequantize_output(raw_outputs[shadow_idx], output_details[shadow_idx])[0, ..., 0]
        exclude_logits = dequantize_output(raw_outputs[exclude_idx], output_details[exclude_idx])[0, ..., 0]
        shadow_prob = 1.0 / (1.0 + np.exp(-shadow_logits))
        exclude_prob = 1.0 / (1.0 + np.exp(-exclude_logits))
        pred = np.zeros_like(y.numpy()[0], dtype=np.uint8)
        pred[(shadow_prob >= args.shadow_threshold) & (exclude_prob < args.exclude_threshold)] = 1
        pred[exclude_prob >= args.exclude_threshold] = 2
        y_true_all.append(y.numpy()[0].astype(np.uint8))
        y_pred_all.append(pred)
        frame_count += 1
    elapsed = time.perf_counter() - start

    report = segmentation_report(np.stack(y_true_all), np.stack(y_pred_all))
    report["desktop_fps"] = frame_count / max(elapsed, 1e-6)
    report["shadow_threshold"] = args.shadow_threshold
    report["exclude_threshold"] = args.exclude_threshold
    report["input_quantization"] = input_detail["quantization"]
    report["output_names"] = [detail["name"] for detail in output_details]
    report["output_quantization"] = [detail["quantization"] for detail in output_details]
    report["shadow_output_index"] = shadow_idx
    report["exclude_output_index"] = exclude_idx
    report["keras_assignment_errors"] = assignment_errors
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
