from __future__ import annotations
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import VGG19

def build_cnn_baseline(input_shape: tuple[int,int,int], num_classes: int, l2_reg: float | None = None) -> keras.Model:
    """Build the custom CNN baseline described in the report.

    Architecture:
      - 4 conv blocks with filters 32/64/128/256
      - Each block: Conv(3x3, BN, ReLU) x2 -> MaxPool(2) -> Dropout
      - Head: GAP -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(num_classes, softmax)

    Args:
        input_shape: (H, W, C)
        num_classes: Number of output classes.
        l2_reg: Optional L2 regularization factor applied to conv & dense kernels.

    Returns:
        A compiled Keras model (uncompiled; caller must compile).
    """
    def L2(): return regularizers.l2(l2_reg) if l2_reg else None

    def block(x, f, drop):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=L2())(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=L2())(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x); x = layers.Dropout(drop)(x); return x

    inp = keras.Input(shape=input_shape)
    x = block(inp, 32, 0.25)
    x = block(x, 64, 0.30)
    x = block(x, 128, 0.35)
    x = block(x, 256, 0.40)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=L2())(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="CustomCNN")

def build_vgg19_model(input_shape: tuple[int,int,int], num_classes: int, dropout: float = 0.3,
                      l2_reg: float = 1e-4) -> keras.Model:
    """Build a VGG19 backbone with a small classification head for transfer learning.

    Args:
        input_shape: (H, W, C)
        num_classes: Number of output classes.
        dropout: Dropout on the head.
        l2_reg: L2 factor for the final dense layer (weight decay proxy).

    Returns:
        A Keras model (with VGG19 frozen) ready for phase-1 training.
    """
    base = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    inp = keras.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax",
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
    return keras.Model(base.input, out, name="VGG19-TL")

def unfreeze_from_block(model: keras.Model, block_idx: int = 4) -> None:
    """Unfreeze VGG19 starting from a specific block (4 or 5).

    Args:
        model: The VGG19-TL model.
        block_idx: 4 to unfreeze `block4_*` and above, or 5 for only the last block.
    """
    trainable = False
    for layer in model.layers:
        if layer.name.startswith(f"block{block_idx}_"):
            trainable = True
        layer.trainable = trainable
