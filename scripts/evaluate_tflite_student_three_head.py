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
    info = np.iinfo(input_detail["dtype"])
    return np.clip(q, info.min, info.max).astype(input_detail["dtype"])


def dequantize_output(y: np.ndarray, output_detail) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def resolve_outputs_by_keras(interpreter, input_detail, output_details, keras_model, ds, max_samples: int):
    permutations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    errors = np.zeros(len(permutations), dtype=np.float64)
    count = 0
    for x, _ in ds.take(max_samples):
        keras_outputs = keras_model(x, training=False)
        if isinstance(keras_outputs, dict):
            keras_outputs = [
                keras_outputs["shadow_logits"].numpy(),
                keras_outputs["exclude_logits"].numpy(),
                keras_outputs["paper_logits"].numpy(),
            ]
        else:
            keras_outputs = [out.numpy() for out in keras_outputs]
        interpreter.set_tensor(input_detail["index"], quantize_input(x.numpy(), input_detail))
        interpreter.invoke()
        outputs = [dequantize_output(interpreter.get_tensor(detail["index"]), detail) for detail in output_details]
        for perm_idx, perm in enumerate(permutations):
            errors[perm_idx] += sum(float(np.mean((outputs[perm[i]] - keras_outputs[i]) ** 2)) for i in range(3))
        count += 1
    if count == 0:
        raise RuntimeError("No samples available to resolve TFLite output order")
    best = int(np.argmin(errors))
    return permutations[best], errors.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--exclude-threshold", type=float, default=0.55)
    parser.add_argument("--paper-threshold", type=float, default=0.25)
    parser.add_argument("--keras-model", default=None)
    parser.add_argument("--order-resolve-samples", type=int, default=16)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    if len(output_details) != 3:
        raise RuntimeError(f"Expected three TFLite outputs, got {len(output_details)}")

    ds = make_dataset(args.data, args.split, batch_size=1, augment=False, shuffle=False)
    assignment_errors = None
    output_order = (0, 1, 2)
    if args.keras_model:
        keras_model = tf.keras.models.load_model(args.keras_model, compile=False)
        output_order, assignment_errors = resolve_outputs_by_keras(
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
        outputs = [dequantize_output(interpreter.get_tensor(detail["index"]), detail) for detail in output_details]
        shadow_logits = outputs[output_order[0]][0, ..., 0]
        exclude_logits = outputs[output_order[1]][0, ..., 0]
        paper_logits = outputs[output_order[2]][0, ..., 0]
        shadow_prob = 1.0 / (1.0 + np.exp(-shadow_logits))
        exclude_prob = 1.0 / (1.0 + np.exp(-exclude_logits))
        paper_prob = 1.0 / (1.0 + np.exp(-paper_logits))
        pred = np.zeros_like(y.numpy()[0], dtype=np.uint8)
        paper_gate = paper_prob >= args.paper_threshold
        pred[(shadow_prob >= args.shadow_threshold) & (exclude_prob < args.exclude_threshold) & paper_gate] = 1
        pred[exclude_prob >= args.exclude_threshold] = 2
        pred[(~paper_gate) & (pred == 0)] = 3
        y_true_all.append(y.numpy()[0].astype(np.uint8))
        y_pred_all.append(pred)
        frame_count += 1
    elapsed = time.perf_counter() - start

    report = segmentation_report(np.stack(y_true_all), np.stack(y_pred_all), num_classes=4)
    report["desktop_fps"] = frame_count / max(elapsed, 1e-6)
    report["shadow_threshold"] = args.shadow_threshold
    report["exclude_threshold"] = args.exclude_threshold
    report["paper_threshold"] = args.paper_threshold
    report["input_quantization"] = input_detail["quantization"]
    report["output_names"] = [detail["name"] for detail in output_details]
    report["output_quantization"] = [detail["quantization"] for detail in output_details]
    report["output_order_shadow_exclude_paper"] = list(output_order)
    report["keras_assignment_errors"] = assignment_errors
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
