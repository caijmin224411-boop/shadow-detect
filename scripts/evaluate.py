#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from shadowcam.dataset import make_dataset
from shadowcam.metrics import segmentation_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="Optional path for the JSON report.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    ds = make_dataset(args.data, args.split, args.batch_size, augment=False, shuffle=False)
    y_true_all = []
    y_pred_all = []
    frame_count = 0
    start = time.perf_counter()
    for x, y in ds:
        logits = model(x, training=False)
        pred = tf.argmax(logits, axis=-1).numpy().astype(np.uint8)
        y_true_all.append(y.numpy().astype(np.uint8))
        y_pred_all.append(pred)
        frame_count += int(x.shape[0])
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    report = segmentation_report(y_true, y_pred)
    report["desktop_fps"] = frame_count / max(elapsed, 1e-6)
    report_json = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
