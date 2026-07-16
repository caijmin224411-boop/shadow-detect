#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from shadowcam.dataset import make_dataset
from shadowcam.model import build_shadow_teacher_unet, build_tiny_shadow_unet, count_parameters


def sparse_dice_loss(y_true, y_logits, num_classes=3):
    y_true = tf.cast(y_true, tf.int32)
    y_prob = tf.nn.softmax(y_logits, axis=-1)
    y_onehot = tf.one_hot(y_true, num_classes)
    axes = [1, 2]
    intersection = tf.reduce_sum(y_prob * y_onehot, axis=axes)
    denom = tf.reduce_sum(y_prob + y_onehot, axis=axes)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(class_weights):
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_logits):
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_logits, from_logits=True)
        sample_weights = tf.gather(weights, tf.cast(y_true, tf.int32))
        ce = tf.reduce_mean(ce * sample_weights)
        return ce + 0.6 * sparse_dice_loss(y_true, y_logits)

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--class-weights", default="1.0,2.5,2.0")
    parser.add_argument(
        "--architecture",
        choices=("tiny", "teacher"),
        default="tiny",
        help="Use the deployable tiny model or the larger annotation-ceiling teacher.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Train on the original frames without synthetic photometric augmentation.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    class_weights = [float(x) for x in args.class_weights.split(",")]

    train_ds = make_dataset(
        args.data,
        "train",
        args.batch_size,
        augment=not args.no_augment,
        shuffle=True,
    )
    val_ds = make_dataset(args.data, "val", args.batch_size, augment=False, shuffle=False)

    builder = build_shadow_teacher_unet if args.architecture == "teacher" else build_tiny_shadow_unet
    model = builder(base_channels=args.base_channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=combined_loss(class_weights),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_acc")],
    )

    metadata = {
        "base_channels": args.base_channels,
        "architecture": args.architecture,
        "parameters": count_parameters(model),
        "input_shape": [96, 96, 1],
        "output_shape": [48, 48, 3],
        "class_weights": class_weights,
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
