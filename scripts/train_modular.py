#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from shadowcam.dataset import make_modular_dataset
from shadowcam.model import build_modular_shadow_exclusion_net, count_parameters


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
    parser.add_argument("--shadow-pos-weight", type=float, default=4.0)
    parser.add_argument("--exclude-pos-weight", type=float, default=10.0)
    parser.add_argument("--shadow-loss-weight", type=float, default=1.0)
    parser.add_argument("--exclude-loss-weight", type=float, default=0.8)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = make_modular_dataset(args.data, "train", args.batch_size, augment=True, shuffle=True)
    val_ds = make_modular_dataset(args.data, "val", args.batch_size, augment=False, shuffle=False)

    model = build_modular_shadow_exclusion_net(base_channels=args.base_channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={
            "shadow_logits": weighted_binary_loss(args.shadow_pos_weight, args.dice_weight),
            "exclude_logits": weighted_binary_loss(args.exclude_pos_weight, args.dice_weight),
        },
        loss_weights={
            "shadow_logits": args.shadow_loss_weight,
            "exclude_logits": args.exclude_loss_weight,
        },
    )

    metadata = {
        "model": "modular_shadow_exclusion_net",
        "base_channels": args.base_channels,
        "parameters": count_parameters(model),
        "input_shape": [96, 96, 1],
        "outputs": {
            "shadow_logits": [48, 48, 1],
            "exclude_logits": [48, 48, 1],
        },
        "shadow_pos_weight": args.shadow_pos_weight,
        "exclude_pos_weight": args.exclude_pos_weight,
        "shadow_loss_weight": args.shadow_loss_weight,
        "exclude_loss_weight": args.exclude_loss_weight,
        "dice_weight": args.dice_weight,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (out / "model_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(out / "history.csv"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(out / "last.keras")


if __name__ == "__main__":
    main()
