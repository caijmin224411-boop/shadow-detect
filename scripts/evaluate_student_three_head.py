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
        return outputs["shadow_logits"], outputs["exclude_logits"], outputs["paper_logits"]
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        by_name = {getattr(t, "name", "").split("/")[0]: t for t in outputs}
        if {"shadow_logits", "exclude_logits", "paper_logits"} <= set(by_name):
            return by_name["shadow_logits"], by_name["exclude_logits"], by_name["paper_logits"]
        return outputs[0], outputs[1], outputs[2]
    raise TypeError(f"Expected three model outputs, got {type(outputs)!r}")


def report_for_thresholds(
    y_true,
    shadow_prob,
    exclude_prob,
    paper_prob,
    shadow_threshold,
    exclude_threshold,
    paper_threshold,
):
    pred = np.zeros_like(y_true, dtype=np.uint8)
    paper_gate = paper_prob >= paper_threshold
    pred[(shadow_prob >= shadow_threshold) & (exclude_prob < exclude_threshold) & paper_gate] = 1
    pred[exclude_prob >= exclude_threshold] = 2
    pred[(~paper_gate) & (pred == 0)] = 3
    report = segmentation_report(y_true, pred, num_classes=4)
    report["shadow_threshold"] = float(shadow_threshold)
    report["exclude_threshold"] = float(exclude_threshold)
    report["paper_threshold"] = float(paper_threshold)
    report["exclude_true_positive_rate"] = float(((y_true == 2) & (pred == 2)).sum() / max((y_true == 2).sum(), 1))
    report["paper_gate_shadow_suppression_rate"] = float(
        ((y_true == 1) & (~paper_gate)).sum() / max((y_true == 1).sum(), 1)
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--exclude-threshold", type=float, default=0.55)
    parser.add_argument("--paper-threshold", type=float, default=0.25)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--min-shadow-recall", type=float, default=0.55)
    parser.add_argument("--max-hand-fpr", type=float, default=0.12)
    parser.add_argument("--max-dark-fpr", type=float, default=0.08)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    ds = make_dataset(args.data, args.split, args.batch_size, augment=False, shuffle=False)
    y_true_all = []
    shadow_prob_all = []
    exclude_prob_all = []
    paper_prob_all = []
    frame_count = 0
    start = time.perf_counter()
    for x, y in ds:
        shadow_logits, exclude_logits, paper_logits = unpack_outputs(model(x, training=False))
        shadow_prob_all.append(tf.nn.sigmoid(shadow_logits).numpy()[..., 0])
        exclude_prob_all.append(tf.nn.sigmoid(exclude_logits).numpy()[..., 0])
        paper_prob_all.append(tf.nn.sigmoid(paper_logits).numpy()[..., 0])
        y_true_all.append(y.numpy().astype(np.uint8))
        frame_count += int(x.shape[0])
    elapsed = time.perf_counter() - start

    y_true = np.concatenate(y_true_all, axis=0)
    shadow_prob = np.concatenate(shadow_prob_all, axis=0)
    exclude_prob = np.concatenate(exclude_prob_all, axis=0)
    paper_prob = np.concatenate(paper_prob_all, axis=0)

    if args.sweep:
        reports = []
        for shadow_threshold in np.arange(0.20, 0.81, 0.05):
            for exclude_threshold in np.arange(0.35, 0.81, 0.05):
                for paper_threshold in (0.10, 0.20, 0.30, 0.40):
                    reports.append(
                        report_for_thresholds(
                            y_true,
                            shadow_prob,
                            exclude_prob,
                            paper_prob,
                            float(shadow_threshold),
                            float(exclude_threshold),
                            float(paper_threshold),
                        )
                    )
        reports.sort(
            key=lambda r: (
                r["hand_false_positive_rate"] <= 0.10,
                r["class_1_recall"],
                r["class_1_iou"],
                -r["dark_nonshadow_false_positive_rate"],
            ),
            reverse=True,
        )
        constrained = [
            r
            for r in reports
            if r["class_1_recall"] >= args.min_shadow_recall
            and r["hand_false_positive_rate"] <= args.max_hand_fpr
            and r["dark_nonshadow_false_positive_rate"] <= args.max_dark_fpr
        ]
        constrained.sort(
            key=lambda r: (
                r["class_1_recall"],
                r["class_1_iou"],
                -r["hand_false_positive_rate"],
                -r["dark_nonshadow_false_positive_rate"],
            ),
            reverse=True,
        )
        print(
            json.dumps(
                {
                    "desktop_fps": frame_count / max(elapsed, 1e-6),
                    "constraints": {
                        "min_shadow_recall": args.min_shadow_recall,
                        "max_hand_fpr": args.max_hand_fpr,
                        "max_dark_fpr": args.max_dark_fpr,
                    },
                    "constrained_count": len(constrained),
                    "constrained_top": constrained[:20],
                    "top": reports[:20],
                },
                indent=2,
            )
        )
        return

    report = report_for_thresholds(
        y_true,
        shadow_prob,
        exclude_prob,
        paper_prob,
        args.shadow_threshold,
        args.exclude_threshold,
        args.paper_threshold,
    )
    report["desktop_fps"] = frame_count / max(elapsed, 1e-6)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
