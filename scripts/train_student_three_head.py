#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from shadowcam.dataset import make_three_head_dataset
from shadowcam.model import (
    build_shadow_exclude_paper_net,
    build_shadow_exclude_paper_net_p4,
    count_parameters,
)


def binary_dice_loss(y_true, logits):
    prob = tf.nn.sigmoid(logits)
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(prob * y_true, axis=axes)
    denom = tf.reduce_sum(prob + y_true, axis=axes)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - tf.reduce_mean(dice)


def weighted_binary_loss(pos_weight: float, dice_weight: float):
    pos_weight_t = tf.constant(pos_weight, dtype=tf.float32)

    def loss(y_true, logits):
        ce = tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(y_true, tf.float32),
            logits=logits,
            pos_weight=pos_weight_t,
        )
        return tf.reduce_mean(ce) + dice_weight * binary_dice_loss(y_true, logits)

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--shadow-pos-weight", type=float, default=5.0)
    parser.add_argument("--exclude-pos-weight", type=float, default=12.0)
    parser.add_argument("--paper-pos-weight", type=float, default=1.0)
    parser.add_argument("--shadow-loss-weight", type=float, default=1.0)
    parser.add_argument("--exclude-loss-weight", type=float, default=0.9)
    parser.add_argument("--paper-loss-weight", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--initial-model", default=None, help="Optional Keras checkpoint to continue training from.")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument(
        "--p4-compatible",
        action="store_true",
        help="Avoid decoder upsampling so exported TFLite fits ESP32-P4 TFLite Micro.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = make_three_head_dataset(args.data, "train", args.batch_size, augment=True, shuffle=True)
    val_ds = make_three_head_dataset(args.data, "val", args.batch_size, augment=False, shuffle=False)

    if args.initial_model:
        model = tf.keras.models.load_model(args.initial_model, compile=False)
    else:
        builder = build_shadow_exclude_paper_net_p4 if args.p4_compatible else build_shadow_exclude_paper_net
        model = builder(base_channels=args.base_channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={
            "shadow_logits": weighted_binary_loss(args.shadow_pos_weight, args.dice_weight),
            "exclude_logits": weighted_binary_loss(args.exclude_pos_weight, args.dice_weight),
            "paper_logits": weighted_binary_loss(args.paper_pos_weight, args.dice_weight),
        },
        loss_weights={
            "shadow_logits": args.shadow_loss_weight,
            "exclude_logits": args.exclude_loss_weight,
            "paper_logits": args.paper_loss_weight,
        },
    )

    metadata = {
        "model": model.name,
        "p4_compatible": args.p4_compatible,
        "base_channels": args.base_channels,
        "initial_model": args.initial_model,
        "parameters": count_parameters(model),
        "input_shape": [96, 96, 1],
        "outputs": {
            "shadow_logits": [48, 48, 1],
            "exclude_logits": [48, 48, 1],
            "paper_logits": [48, 48, 1],
        },
        "loss_weights": {
            "shadow_logits": args.shadow_loss_weight,
            "exclude_logits": args.exclude_loss_weight,
            "paper_logits": args.paper_loss_weight,
        },
        "positive_weights": {
            "shadow_logits": args.shadow_pos_weight,
            "exclude_logits": args.exclude_pos_weight,
            "paper_logits": args.paper_pos_weight,
        },
        "dice_weight": args.dice_weight,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (out / "model_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(out / "history.csv", append=bool(args.initial_model)),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=args.verbose)
    model.save(out / "last.keras")


if __name__ == "__main__":
    main()
