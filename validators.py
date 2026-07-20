"""CT scan input validation (ported from notebook Cell 19B)."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image as PILImage

PROFILE_VERSION = 4

CT_GATE_MODE: str | None = None
CT_GATE_CLF = None
CT_REF_EMBEDDINGS: np.ndarray | None = None
CT_DIST_THRESHOLD: float | None = None
CT_EDGE_MEAN: float | None = None
CT_EDGE_STD: float | None = None

_generic_backbone: nn.Module | None = None
_generic_tf: T.Compose | None = None
_device: torch.device | None = None


def _ensure_backbone(device: torch.device) -> None:
    global _generic_backbone, _generic_tf, _device
    if _generic_backbone is not None and _device == device:
        return
    _device = device
    _generic_backbone = tv_models.resnet18(weights="IMAGENET1K_V1")
    _generic_backbone.fc = nn.Identity()
    _generic_backbone.eval().to(device)
    for param in _generic_backbone.parameters():
        param.requires_grad = False
    _generic_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _generic_embed(pil_img: PILImage.Image, device: torch.device) -> np.ndarray:
    _ensure_backbone(device)
    x = _generic_tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _generic_backbone(x).squeeze(0).cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def _edge_density(pil_img: PILImage.Image, size: int = 224) -> float:
    gray = np.array(pil_img.convert("L").resize((size, size)), dtype=np.float32)
    gy, gx = np.gradient(gray)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def _knn_distance(query: np.ndarray, reference: np.ndarray, k: int = 3) -> float:
    sims = reference @ query
    dists = np.sort(1.0 - sims)
    k = min(k, len(dists))
    return float(dists[k - 1])


def load_ct_validator(checkpoint_dir: str | Path, device: torch.device) -> None:
    """Load cached CT validator profile if available."""
    global CT_GATE_MODE, CT_GATE_CLF, CT_REF_EMBEDDINGS
    global CT_DIST_THRESHOLD, CT_EDGE_MEAN, CT_EDGE_STD

    profile_path = Path(checkpoint_dir) / "ct_ref_profile.joblib"
    if not profile_path.exists():
        return

    profile = joblib.load(profile_path)
    if profile.get("version") != PROFILE_VERSION:
        return

    CT_GATE_MODE = profile["mode"]
    CT_GATE_CLF = profile.get("clf")
    CT_REF_EMBEDDINGS = profile.get("embeddings")
    CT_DIST_THRESHOLD = profile.get("threshold")
    CT_EDGE_MEAN = profile["edge_mean"]
    CT_EDGE_STD = profile["edge_std"]


def validate_ct_image(pil_img: PILImage.Image | None, device: torch.device) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if pil_img is None:
        return False, "No image data received."

    w, h = pil_img.size
    if w < 32 or h < 32:
        return False, "Image too small to be a CT scan."

    rgb = pil_img.convert("RGB")
    arr = np.array(rgb).astype(np.float32)
    chan_divergence = (
        np.abs(arr[..., 0] - arr[..., 1]).mean()
        + np.abs(arr[..., 1] - arr[..., 2]).mean()
    )
    if chan_divergence > 8.0:
        return False, "Image is in colour — not a valid greyscale CT scan."

    gray = np.array(pil_img.convert("L")).astype(np.float32)
    if gray.std() < 5.0:
        return False, "Image has near-uniform intensity — not a valid CT scan."

    if CT_EDGE_MEAN is not None and CT_EDGE_STD is not None:
        edge_val = _edge_density(pil_img)
        z = abs(edge_val - CT_EDGE_MEAN) / CT_EDGE_STD
        if z > 4.0:
            return False, f"Image texture does not match chest CT scans (edge z-score={z:.1f})."

    if CT_GATE_MODE == "supervised" and CT_GATE_CLF is not None:
        emb = _generic_embed(pil_img, device)
        proba = CT_GATE_CLF.predict_proba(emb.reshape(1, -1))[0, 1]
        if proba < 0.5:
            return (
                False,
                f"Classified as non-chest-CT by the supervised gate (P(chest CT)={proba:.2f}).",
            )
    elif CT_GATE_MODE == "unsupervised" and CT_REF_EMBEDDINGS is not None:
        emb = _generic_embed(pil_img, device)
        dist = _knn_distance(emb, CT_REF_EMBEDDINGS)
        if CT_DIST_THRESHOLD is not None and dist > CT_DIST_THRESHOLD:
            return (
                False,
                f"Image does not match the learned appearance of a chest CT "
                f"(distance={dist:.3f}, limit={CT_DIST_THRESHOLD:.3f}).",
            )

    return True, "Valid CT scan."
