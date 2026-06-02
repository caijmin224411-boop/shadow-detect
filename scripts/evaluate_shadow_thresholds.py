#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

from shadowcam.dataset import make_dataset


def binary_metrics(y_true_class, shadow_score, threshold: float):
    y_true_shadow = y_true_class == 1
    y_pred_shadow = shadow_score >= threshold
    tp = np.logical_and(y_true_shadow, y_pred_shadow).sum()
    fp = np.logical_and(~y_true_shadow, y_pred_shadow).sum()
    fn = np.logical_and(y_true_shadow, ~y_pred_shadow).sum()
    union = tp + fp + fn
    hand_pixels = y_true_class == 2
    dark_nonshadow = y_true_class != 1
    return {
        "threshold": threshold,
        "shadow_iou": float(tp / union) if union else 1.0,
        "shadow_recall": float(tp / (tp + fn)) if (tp + fn) else 1.0,
        "shadow_precision": float(tp / (tp + fp)) if (tp + fp) else 1.0,
        "shadow_pred_ratio": float(y_pred_shadow.sum() / max(y_pred_shadow.size, 1)),
        "hand_false_positive_rate": float(np.logical_and(hand_pixels, y_pred_shadow).sum() / max(hand_pixels.sum(), 1)),
        "dark_nonshadow_false_positive_rate": float(np.logical_and(dark_nonshadow, y_pred_shadow).sum() / max(dark_nonshadow.sum(), 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    ds = make_dataset(args.data, args.split, args.batch_size, augment=False, shuffle=False)
    y_true_all = []
    shadow_score_all = []
    for x, y in ds:
        logits = model(x, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        y_true_all.append(y.numpy().astype(np.uint8))
        shadow_score_all.append(probs[..., 1])

    y_true = np.concatenate(y_true_all, axis=0)
    shadow_score = np.concatenate(shadow_score_all, axis=0)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    report = [binary_metrics(y_true, shadow_score, t) for t in thresholds]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

