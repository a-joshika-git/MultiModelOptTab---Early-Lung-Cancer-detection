"""Load ensemble checkpoint and run predictions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage

from model import GradCamWrapper, MultiModalOptTab
from validators import load_ct_validator, validate_ct_image


@dataclass
class PredictionResult:
    prediction: str
    calibrated_prob: float
    raw_prob: float
    threshold: float
    uncertainty: float
    member_probs: list[float]
    flag_for_review: bool
    gradcam_image: np.ndarray | None
    agreement: float | None
    validation_message: str


class MultiModalPredictor:
    def __init__(self, checkpoint_dir: str | Path | None = None):
        self.checkpoint_dir = Path(
            checkpoint_dir or os.environ.get("CHECKPOINT_DIR", "checkpoint")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self.ensemble_models: list[MultiModalOptTab] = []
        self.config: dict = {}
        self.qt = None
        self.le_gender = None
        self.oe = None
        self.platt = None
        self.val_transforms = None
        self.tab_input_dim = 0
        self.best_threshold = 0.5

    def load(self) -> None:
        if self._loaded:
            return

        required = [
            "config.joblib",
            "qt.joblib",
            "le_gender.joblib",
            "oe.joblib",
            "platt.joblib",
            "ensemble_member_1.pth",
        ]
        missing = [name for name in required if not (self.checkpoint_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint files in "
                f"{self.checkpoint_dir.resolve()}: {', '.join(missing)}. "
                "Copy the files saved by notebook Cell 18 from Google Drive "
                "into the checkpoint/ folder."
            )

        self.config = joblib.load(self.checkpoint_dir / "config.joblib")
        self.tab_input_dim = int(self.config["TAB_INPUT_DIM"])
        self.best_threshold = float(self.config["BEST_THRESHOLD"])
        ensemble_configs = self.config["ENSEMBLE_CONFIGS"]
        img_size = int(self.config.get("IMG_SIZE", 224))

        self.qt = joblib.load(self.checkpoint_dir / "qt.joblib")
        self.le_gender = joblib.load(self.checkpoint_dir / "le_gender.joblib")
        self.oe = joblib.load(self.checkpoint_dir / "oe.joblib")
        self.platt = joblib.load(self.checkpoint_dir / "platt.joblib")

        self.val_transforms = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )

        self.ensemble_models = []
        for i, cfg in enumerate(ensemble_configs):
            model = MultiModalOptTab(
                tab_input_dim=self.tab_input_dim,
                n_heads=cfg["n_heads"],
                n_transformer_layers=cfg["n_layers"],
                dropout_rate=cfg["dropout"],
            ).to(self.device)
            weights_path = self.checkpoint_dir / f"ensemble_member_{i + 1}.pth"
            try:
                state = torch.load(
                    weights_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.ensemble_models.append(model)

        load_ct_validator(self.checkpoint_dir, self.device)
        self._loaded = True

    def _build_tabular_tensor(self, age: int, gender: str, smoking: str) -> torch.Tensor:
        gender_enc = self.le_gender.transform([gender])[0]
        smoking_enc = self.oe.transform([[smoking]])[0, 0]
        row_raw = np.array(
            [
                [
                    age,
                    gender_enc,
                    smoking_enc,
                    age * smoking_enc,
                    (age / 100) + (smoking_enc / 2),
                    age**2,
                    smoking_enc**2,
                    age * gender_enc,
                    np.log1p(age),
                    smoking_enc * gender_enc,
                ]
            ],
            dtype=np.float32,
        )
        row_scaled = self.qt.transform(row_raw)
        return torch.tensor(row_scaled, dtype=torch.float32).to(self.device)

    def predict(
        self,
        image: PILImage.Image,
        age: int,
        gender: str,
        smoking: str,
        mc_passes: int = 20,
        threshold: float | None = None,
        include_gradcam: bool = True,
    ) -> PredictionResult:
        self.load()
        threshold = self.best_threshold if threshold is None else threshold

        is_valid, reason = validate_ct_image(image, self.device)
        if not is_valid:
            return PredictionResult(
                prediction="INVALID",
                calibrated_prob=0.0,
                raw_prob=0.0,
                threshold=threshold,
                uncertainty=0.0,
                member_probs=[],
                flag_for_review=False,
                gradcam_image=None,
                agreement=None,
                validation_message=reason,
            )

        pil_gray = image.convert("L")
        img_np = np.array(pil_gray)
        img_t = (
            self.val_transforms(image=img_np)["image"]
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        tab_t = self._build_tabular_tensor(age, gender, smoking)

        all_member_probs: list[float] = []
        for model in self.ensemble_models:
            pass_probs: list[float] = []
            with torch.no_grad():
                for _ in range(mc_passes):
                    logits = model.forward_once(img_t, tab_t)
                    pass_probs.append(F.softmax(logits, dim=-1)[0, 1].item())
            all_member_probs.append(float(np.mean(pass_probs)))

        raw_prob = float(np.mean(all_member_probs))
        uncertainty = float(np.var(all_member_probs))
        calibrated_prob = float(self.platt.predict_proba([[raw_prob]])[0, 1])
        prediction = "CANCER" if calibrated_prob > threshold else "NO CANCER"
        flag_for_review = uncertainty > 0.05

        gradcam_image = None
        agreement = None
        if include_gradcam:
            gradcam_image, agreement = self._make_gradcam(img_t)

        return PredictionResult(
            prediction=prediction,
            calibrated_prob=calibrated_prob,
            raw_prob=raw_prob,
            threshold=threshold,
            uncertainty=uncertainty,
            member_probs=all_member_probs,
            flag_for_review=flag_for_review,
            gradcam_image=gradcam_image,
            agreement=agreement,
            validation_message=reason,
        )

    def _make_gradcam(self, img_t: torch.Tensor) -> tuple[np.ndarray | None, float | None]:
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            raw_disp = img_t.squeeze().cpu().numpy()
            raw_disp = (raw_disp - raw_disp.min()) / (raw_disp.max() - raw_disp.min() + 1e-8)
            rgb_disp = np.stack([raw_disp] * 3, axis=-1).astype(np.float32)

            member_cams = []
            for model in self.ensemble_models:
                wrapper = GradCamWrapper(model, self.tab_input_dim)
                cam = GradCAMPlusPlus(
                    model=wrapper, target_layers=[model.img_encoder.features[-1]]
                )
                saliency = cam(
                    input_tensor=img_t, targets=[ClassifierOutputTarget(1)]
                )[0]
                member_cams.append(saliency)

            member_cams_arr = np.stack(member_cams)
            ensemble_cam = member_cams_arr.mean(axis=0)
            flat = member_cams_arr.reshape(len(member_cams_arr), -1)
            norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
            sim_matrix = norm @ norm.T
            agreement = float(sim_matrix[np.triu_indices(len(member_cams_arr), k=1)].mean())
            return show_cam_on_image(rgb_disp, ensemble_cam, use_rgb=True), agreement
        except Exception:
            return None, None
