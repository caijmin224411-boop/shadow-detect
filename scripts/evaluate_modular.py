#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from shadowcam.dataset import make_dataset
from shadowcam.metrics import segmentation_report


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["shadow_logits"], outputs["exclude_logits"]
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        names = [getattr(t, "name", "") for t in outputs]
        if names[0].startswith("exclude"):
            return outputs[1], outputs[0]
        return outputs[0], outputs[1]
    raise TypeError(f"Expected two model outputs, got {type(outputs)!r}")


def report_for_thresholds(y_true, shadow_prob, exclude_prob, shadow_threshold, exclude_threshold):
    pred = np.zeros_like(y_true, dtype=np.uint8)
    pred[(shadow_prob >= shadow_threshold) & (exclude_prob < exclude_threshold)] = 1
    pred[exclude_prob >= exclude_threshold] = 2
    report = segmentation_report(y_true, pred)
    report["shadow_threshold"] = float(shadow_threshold)
    report["exclude_threshold"] = float(exclude_threshold)
    report["exclude_true_positive_rate"] = float(((y_true == 2) & (pred == 2)).sum() / max((y_true == 2).sum(), 1))
    report["shadow_suppressed_by_exclude_rate"] = float(
        ((y_true == 1) & (exclude_prob >= exclude_threshold)).sum() / max((y_true == 1).sum(), 1)
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--exclude-threshold", type=float, default=0.50)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    ds = make_dataset(args.data, args.split, args.batch_size, augment=False, shuffle=False)
    y_true_all = []
    shadow_prob_all = []
    exclude_prob_all = []
    frame_count = 0
    start = time.perf_counter()
    for x, y in ds:
        shadow_logits, exclude_logits = unpack_outputs(model(x, training=False))
        shadow_prob_all.append(tf.nn.sigmoid(shadow_logits).numpy()[..., 0])
        exclude_prob_all.append(tf.nn.sigmoid(exclude_logits).numpy()[..., 0])
        y_true_all.append(y.numpy().astype(np.uint8))
        frame_count += int(x.shape[0])
    elapsed = time.perf_counter() - start

    y_true = np.concatenate(y_true_all, axis=0)
    shadow_prob = np.concatenate(shadow_prob_all, axis=0)
    exclude_prob = np.concatenate(exclude_prob_all, axis=0)

    if args.sweep:
        reports = []
        for shadow_threshold in np.arange(0.20, 0.76, 0.05):
            for exclude_threshold in np.arange(0.35, 0.76, 0.05):
                reports.append(
                    report_for_thresholds(
                        y_true,
                        shadow_prob,
                        exclude_prob,
                        float(shadow_threshold),
                        float(exclude_threshold),
                    )
                )
        reports.sort(
            key=lambda r: (
                r["hand_false_positive_rate"] <= 0.05,
                r["class_1_recall"],
                r["class_1_iou"],
                -r["hand_false_positive_rate"],
            ),
            reverse=True,
        )
        print(json.dumps({"desktop_fps": frame_count / max(elapsed, 1e-6), "top": reports[:20]}, indent=2))
        return

    report = report_for_thresholds(
        y_true,
        shadow_prob,
        exclude_prob,
        args.shadow_threshold,
        args.exclude_threshold,
    )
    report["desktop_fps"] = frame_count / max(elapsed, 1e-6)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
