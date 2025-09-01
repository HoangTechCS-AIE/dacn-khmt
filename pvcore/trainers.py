from __future__ import annotations
import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import SEED, IMG_SIZE, VAL_SPLIT, TEST_SPLIT, setup_seeds
from .data import list_image_files, split_items, compute_class_weights, make_dataset, aug_cnn, aug_vgg
from .models import build_cnn_baseline, build_vgg19_model, unfreeze_from_block

def train_cnn(work_dir: str, data_root: str, batch_size: int = 32, epochs: int = 30,
              lr: float = 1e-4, l2: float = 0.0) -> dict:
    """Train the custom CNN baseline end-to-end.

    Saves the best model (by val_loss) to ``work_dir/cnn_best.h5``
    and writes final metrics to ``work_dir/result.json``.

    Args:
        work_dir: Output directory for artifacts.
        data_root: Dataset root folder.
        batch_size: Batch size.
        epochs: Maximum epochs (EarlyStopping applies).
        lr: Adam learning rate.
        l2: Optional L2 for conv/dense kernels.

    Returns:
        A dictionary with final validation/test loss and accuracy.
    """
    os.makedirs(work_dir, exist_ok=True)
    setup_seeds(SEED)
    class_names, items = list_image_files(data_root)
    num_classes = len(class_names)
    train_items, val_items, test_items = split_items(items, VAL_SPLIT, TEST_SPLIT, seed=SEED)
    class_weight, _ = compute_class_weights(train_items, num_classes)

    rescale = keras.layers.Rescaling(1./255.)
    train_ds = make_dataset(train_items, batch_size, IMG_SIZE, True, aug_cnn(), rescale)
    val_ds   = make_dataset(val_items, batch_size, IMG_SIZE, False, None, rescale)
    test_ds  = make_dataset(test_items, batch_size, IMG_SIZE, False, None, rescale)

    model = build_cnn_baseline((*IMG_SIZE,3), num_classes, l2_reg=(l2 if l2>0 else None))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        keras.callbacks.ModelCheckpoint(os.path.join(work_dir,"cnn_best.h5"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=cbs, verbose=1)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    out = {"val_loss": float(val_loss), "val_acc": float(val_acc),
           "test_loss": float(test_loss), "test_acc": float(test_acc)}
    json.dump(out, open(os.path.join(work_dir,"result.json"),"w"), indent=2)
    return out

def train_vgg19(work_dir: str, data_root: str, batch_size: int = 32, phase1_epochs: int = 8,
                phase2_epochs: int = 30, fine_tune_from_block: int = 4, lr1: float = 1e-3,
                lr2: float = 1e-5, dropout: float = 0.3, weight_decay: float = 1e-4) -> dict:
    """Two-phase transfer learning with VGG19.

    Phase 1: train head with backbone frozen.
    Phase 2: unfreeze from the specified block and fine-tune with small LR.

    Saves the best fine-tuned model to ``work_dir/vgg19_best.h5`` and metrics to ``result.json``.

    Args:
        work_dir: Output directory for artifacts.
        data_root: Dataset root.
        batch_size: Batch size.
        phase1_epochs: Freeze phase epochs.
        phase2_epochs: Fine-tune phase epochs.
        fine_tune_from_block: 4 or 5.
        lr1: Learning rate for phase 1.
        lr2: Learning rate for phase 2.
        dropout: Dropout on the head.
        weight_decay: L2 for the final dense layer.

    Returns:
        A dictionary with final validation/test metrics.
    """
    os.makedirs(work_dir, exist_ok=True)
    setup_seeds(SEED)
    class_names, items = list_image_files(data_root)
    num_classes = len(class_names)
    train_items, val_items, test_items = split_items(items, VAL_SPLIT, TEST_SPLIT, seed=SEED)
    class_weight, _ = compute_class_weights(train_items, num_classes)

    vgg_pre = lambda x: tf.keras.applications.vgg19.preprocess_input(x)
    train_ds = make_dataset(train_items, batch_size, IMG_SIZE, True, aug_vgg(), vgg_pre)
    val_ds   = make_dataset(val_items, batch_size, IMG_SIZE, False, None, vgg_pre)
    test_ds  = make_dataset(test_items, batch_size, IMG_SIZE, False, None, vgg_pre)

    model = build_vgg19_model((*IMG_SIZE,3), num_classes, dropout=dropout, l2_reg=weight_decay)
    model.compile(optimizer=keras.optimizers.Adam(lr1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs, class_weight=class_weight, verbose=1)

    unfreeze_from_block(model, block_idx=fine_tune_from_block)
    model.compile(optimizer=keras.optimizers.Adam(lr2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        keras.callbacks.ModelCheckpoint(os.path.join(work_dir,"vgg19_best.h5"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs, class_weight=class_weight, callbacks=cbs, verbose=1)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    out = {"val_loss": float(val_loss), "val_acc": float(val_acc),
           "test_loss": float(test_loss), "test_acc": float(test_acc)}
    json.dump(out, open(os.path.join(work_dir,"result.json"),"w"), indent=2)
    return out

def infer_image(model_path: str, image_path: str, img_size: tuple[int,int] = (224,224), topk: int = 5) -> list[dict]:
    """Run single-image inference using a saved Keras model.

    If the model filename contains ``vgg19``, ImageNet preprocessing is applied;
    otherwise a 1/255 rescaling is used.

    Args:
        model_path: Path to ``.h5`` model file.
        image_path: Path to input image.
        img_size: Target size (H, W) for resizing.
        topk: Number of top predictions to return.

    Returns:
        List of dicts: ``[{"class_index": int, "prob": float}, ...]`` sorted by probability.
    """
    model = keras.models.load_model(model_path)
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    if "vgg19" in os.path.basename(model_path).lower():
        x = tf.keras.applications.vgg19.preprocess_input(img)
    else:
        x = img / 255.0
    probs = model.predict(tf.expand_dims(x, 0), verbose=0)[0]
    idx = np.argsort(-probs)[:topk]
    return [{"class_index": int(i), "prob": float(probs[i])} for i in idx]
