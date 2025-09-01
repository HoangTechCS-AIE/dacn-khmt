from __future__ import annotations
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

def list_image_files(root_dir: str) -> tuple[list[str], list[tuple[str, int]]]:
    """List (image_path, class_index) pairs from a directory-of-directories dataset.

    Args:
        root_dir: Root folder where each subdirectory corresponds to a class.

    Returns:
        A pair (class_names, items) where:
          - class_names: list of class folder names sorted alphabetically
          - items: list of (absolute_path, class_index) tuples
    """
    root = pathlib.Path(root_dir)
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    items: list[tuple[str, int]] = []
    for ci, cname in enumerate(class_names):
        for f in (root / cname).glob("**/*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                items.append((str(f), ci))
    return class_names, items

def split_items(items: list[tuple[str, int]], val_split: float = 0.15,
                test_split: float = 0.10, seed: int = 42) -> tuple[list, list, list]:
    """Split items into train/val/test with a deterministic shuffle.

    Args:
        items: List of (path, class_index) pairs.
        val_split: Fraction reserved for validation+test together.
        test_split: Fraction of the validation set to be used as the test subset.
        seed: RNG seed for shuffling.

    Returns:
        (train_items, val_items, test_items)
    """
    rng = np.random.default_rng(seed)
    items = list(items)  # copy
    rng.shuffle(items)
    n = len(items)
    n_val = int(n * val_split)
    train_items = items[: n - n_val]
    val_items = items[n - n_val :]
    n_test = int(len(val_items) * test_split)
    test_items = val_items[:n_test]
    val_items = val_items[n_test:]
    return train_items, val_items, test_items

def compute_class_weights(pairs: list[tuple[str, int]], num_classes: int) -> tuple[dict[int, float], np.ndarray]:
    """Compute inverse-frequency class weights.

    Args:
        pairs: (path, class_index) pairs for the training split.
        num_classes: Number of classes.

    Returns:
        (weights, counts) where `weights` is a dict usable as Keras `class_weight`.
    """
    counts = np.zeros(num_classes, dtype=int)
    for _, y in pairs:
        counts[y] += 1
    total = counts.sum()
    weights = {i: (total / (num_classes * max(1, counts[i]))) for i in range(num_classes)}
    return weights, counts

def _load_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize an image; keep in float32 [0..255]."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img.set_shape((*img_size, 3))
    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)

def make_dataset(pairs: list[tuple[str, int]], batch_size: int, img_size: tuple[int, int],
                 shuffle: bool = False, aug: keras.Model | None = None, norm=None) -> tf.data.Dataset:
    """Create a performant tf.data pipeline.

    Args:
        pairs: (path, class_index) samples.
        batch_size: Batch size.
        img_size: Resize target (H, W).
        shuffle: Whether to shuffle.
        aug: Keras Sequential augmentation model (optional).
        norm: Normalization/preprocess function mapping tensor->tensor (optional).

    Returns:
        A batched, prefetched tf.data.Dataset of (image, label).
    """
    ds = tf.data.Dataset.from_tensor_slices(([p for p,_ in pairs], [y for _,y in pairs]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _load_image(p,y,img_size), num_parallel_calls=AUTOTUNE)
    if aug is not None:
        ds = ds.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    if norm is not None:
        ds = ds.map(lambda x,y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def aug_cnn() -> keras.Sequential:
    """Augmentations for the custom CNN baseline (with mild color jitter)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.Lambda(lambda x: tf.image.random_brightness(x, 0.08)),
        layers.Lambda(lambda x: tf.image.random_contrast(x, 0.9, 1.1)),
    ], name="aug_cnn")

def aug_vgg() -> keras.Sequential:
    """Augmentations for VGG19 transfer learning (keep color jitter minimal)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.Lambda(lambda x: tf.image.random_contrast(x, 0.95, 1.05)),
    ], name="aug_vgg")
