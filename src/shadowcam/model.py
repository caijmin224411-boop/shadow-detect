from __future__ import annotations

import tensorflow as tf

from .config import INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES


def _conv_bn_relu(x, channels: int, name: str):
    x = tf.keras.layers.Conv2D(
        channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)


def _ds_block(x, channels: int, name: str, stride: int = 1):
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_dw_relu")(x)
    return _conv_bn_relu(x, channels, name=f"{name}_proj")


def build_tiny_shadow_unet(
    input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1),
    num_classes: int = NUM_CLASSES,
    base_channels: int = 8,
) -> tf.keras.Model:
    """Build a TFLite-Micro-friendly depthwise U-Net.

    The network emits 48x48 logits for a 96x96 grayscale input.
    """

    inputs = tf.keras.Input(shape=input_shape, name="image")

    x0 = _conv_bn_relu(inputs, base_channels, "stem")
    x0 = _ds_block(x0, base_channels, "enc0_a")

    x1 = _ds_block(x0, base_channels * 2, "enc1_down", stride=2)  # 48x48
    x1 = _ds_block(x1, base_channels * 2, "enc1_a")

    x2 = _ds_block(x1, base_channels * 3, "enc2_down", stride=2)  # 24x24
    x2 = _ds_block(x2, base_channels * 3, "enc2_a")

    x3 = _ds_block(x2, base_channels * 4, "enc3_down", stride=2)  # 12x12
    x3 = _ds_block(x3, base_channels * 4, "enc3_a")

    u2 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="up2")(x3)
    u2 = tf.keras.layers.Concatenate(name="skip2")([u2, x2])
    u2 = _ds_block(u2, base_channels * 3, "dec2")

    u1 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="up1")(u2)
    u1 = tf.keras.layers.Concatenate(name="skip1")([u1, x1])
    u1 = _ds_block(u1, base_channels * 2, "dec1")

    logits = tf.keras.layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        name="logits",
    )(u1)

    return tf.keras.Model(inputs=inputs, outputs=logits, name="tiny_shadow_unet")


def _tiny_feature_pyramid(inputs, base_channels: int):
    x0 = _conv_bn_relu(inputs, base_channels, "stem")
    x0 = _ds_block(x0, base_channels, "enc0_a")

    x1 = _ds_block(x0, base_channels * 2, "enc1_down", stride=2)  # 48x48
    x1 = _ds_block(x1, base_channels * 2, "enc1_a")

    x2 = _ds_block(x1, base_channels * 3, "enc2_down", stride=2)  # 24x24
    x2 = _ds_block(x2, base_channels * 3, "enc2_a")

    x3 = _ds_block(x2, base_channels * 4, "enc3_down", stride=2)  # 12x12
    x3 = _ds_block(x3, base_channels * 4, "enc3_a")

    u2 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="up2")(x3)
    u2 = tf.keras.layers.Concatenate(name="skip2")([u2, x2])
    u2 = _ds_block(u2, base_channels * 3, "dec2")

    u1 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="up1")(u2)
    u1 = tf.keras.layers.Concatenate(name="skip1")([u1, x1])
    return _ds_block(u1, base_channels * 2, "dec1")


def build_modular_shadow_exclusion_net(
    input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1),
    base_channels: int = 8,
) -> tf.keras.Model:
    """Build a two-head shadow/exclusion model for tiny-device deployment.

    The first head predicts paper shadow. The second head predicts pixels that
    should suppress shadow output, such as hands or dark objects. Keeping the
    heads independent makes the ESP32 post-processing modular while sharing the
    expensive feature extractor.
    """

    inputs = tf.keras.Input(shape=input_shape, name="image")
    features = _tiny_feature_pyramid(inputs, base_channels)
    shadow_logits = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        name="shadow_logits",
    )(features)
    exclude_logits = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        name="exclude_logits",
    )(features)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[shadow_logits, exclude_logits],
        name="modular_shadow_exclusion_net",
    )


def count_parameters(model: tf.keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
