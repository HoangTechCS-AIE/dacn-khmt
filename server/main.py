from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os, shutil

from pvcore.trainers import train_cnn, train_vgg19, infer_image
from pvcore.config import auto_data_root
from pvcore.data import list_image_files

app = FastAPI(title="PlantVillage Trainer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:3000", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict:
    """Health check endpoint.

    Returns:
        A small JSON object indicating the server is up.
    """
    return {"ok": True}

@app.post("/train/cnn")
def start_train_cnn(
    data_root: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    l2: float = 0.0,
) -> dict:
    """Synchronous training for the custom CNN.

    Train and return final metrics (also saved to ``runs/cnn``).

    Args:
        data_root: Optional dataset root. If not provided, auto-detected.
        batch_size: Batch size.
        epochs: Max epochs.
        lr: Learning rate.
        l2: L2 regularization for kernels.

    Returns:
        Training metrics dictionary.
    """
    dr = data_root or auto_data_root()
    os.makedirs("runs", exist_ok=True)
    return train_cnn(work_dir="runs/cnn", data_root=dr, batch_size=batch_size,
                     epochs=epochs, lr=lr, l2=l2)

@app.post("/train/vgg19")
def start_train_vgg19(
    data_root: Optional[str] = None,
    batch_size: int = 32,
    phase1_epochs: int = 8,
    phase2_epochs: int = 30,
    fine_tune_from_block: int = 4,
    lr1: float = 1e-3,
    lr2: float = 1e-5,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
) -> dict:
    """Synchronous two-phase VGG19 training.

    Args mirror the trainer function; artifacts saved to ``runs/vgg19``.

    Returns:
        Training metrics dictionary.
    """
    dr = data_root or auto_data_root()
    os.makedirs("runs", exist_ok=True)
    return train_vgg19(work_dir="runs/vgg19", data_root=dr, batch_size=batch_size,
                       phase1_epochs=phase1_epochs, phase2_epochs=phase2_epochs,
                       fine_tune_from_block=fine_tune_from_block, lr1=lr1, lr2=lr2,
                       dropout=dropout, weight_decay=weight_decay)

@app.get("/labels")
def labels(data_root: Optional[str] = None) -> dict:
    """Return class names as used during training/organization.

    Args:
        data_root: Optional root. If omitted, auto-detect.

    Returns:
        ``{"labels": [str, ...]}`` in the same order used for indices.
    """
    dr = data_root or auto_data_root()
    class_names, _ = list_image_files(dr)
    return {"labels": class_names}

@app.post("/infer")
async def infer(
    model_path: str = Form(...),
    topk: int = Form(5),
    data_root: Optional[str] = Form(None),
    image_file: UploadFile = File(None),
    image_path: Optional[str] = Form(None),
) -> dict:
    """Run inference using either an uploaded image or a server-side `image_path`.

    On success, the server responds with a list of predictions sorted by probability.
    If labels are known, the response includes a ``label`` for each item.

    Args:
        model_path: Server-side path to a saved `.h5` model (e.g., `runs/vgg19/vgg19_best.h5`).
        topk: Number of predictions to return.
        data_root: Optional dataset root to recover label names.
        image_file: Uploaded file (multipart/form-data).
        image_path: Server-accessible image path (alternative to file upload).

    Returns:
        ``{"predictions": [{"class_index": int, "prob": float, "label": str?}, ...]}``
    """
    if image_file is None and not image_path:
        raise HTTPException(400, "Cần gửi image_file (upload) hoặc image_path")

    tmp_path = None
    try:
        if image_file is not None:
            os.makedirs("runs", exist_ok=True)
            tmp_path = os.path.join("runs", f"tmp_{image_file.filename.replace('/', '_')}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            use_path = tmp_path
        else:
            use_path = image_path

        preds = infer_image(model_path=model_path, image_path=use_path, topk=topk)

        # Attach label names if available
        labels = None
        try:
            dr = data_root or auto_data_root()
            labels, _ = list_image_files(dr)
        except Exception:
            pass

        if labels:
            for p in preds:
                p["label"] = labels[p["class_index"]]

        return {"predictions": preds}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
