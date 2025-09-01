from __future__ import annotations
import os
import pathlib
import random
import numpy as np
import tensorflow as tf

# def
SEED: int = 42
IMG_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.10


DATA_DIRS: list[str] = [
    "/shared/data/plantvillage",
    "/shared/data/plantvillage-dataset/PlantVillage",
    "/shared/data/plantvillage-dataset/color",
    "/shared/data/plantvillage-color",
]

def setup_seeds(seed: int = SEED) -> None:
    """Set Python/NumPy/TensorFlow RNG seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def enable_mixed_precision(flag: bool) -> None:
    """Enable or disable TensorFlow mixed precision policy.

    Mixed precision can speed up training on modern GPUs while reducing VRAM.
    On CPU-only environments it has no benefit.

    Args:
        flag: If True, sets global policy to ``mixed_float16``.
    """
    if flag:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

def auto_data_root() -> str:
    """Infer a plausible dataset root folder.

    Resolution order:
      1. `DATA_ROOT` environment variable (if exists and valid)
      2. Known Kaggle paths in :data:`DATA_DIRS`
      3. Local ``./data`` as a conventional fallback

    Returns:
        The resolved directory path.

    Raises:
        FileNotFoundError: If no suitable dataset directory was found.
    """
    env = os.getenv("DATA_ROOT")
    if env and os.path.isdir(env):
        return env

    for d in DATA_DIRS:
        if os.path.isdir(d):
            return d

    local = pathlib.Path("data")
    if local.exists():
        return str(local)

    raise FileNotFoundError(
        "Không tìm thấy PlantVillage. Đặt biến môi trường DATA_ROOT hoặc tạo ./data."
    )
