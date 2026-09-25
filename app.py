"""Gradio web app for MultiModalOptTab lung cancer risk prediction."""

from __future__ import annotations

import os

import gradio as gr
import numpy as np
from PIL import Image

from inference import MultiModalPredictor, PredictionResult

predictor = MultiModalPredictor()
MODEL_STATUS = "Loading model..."


def _format_result(result: PredictionResult) -> tuple[str, str, np.ndarray | None]:
    if result.prediction == "INVALID":
        return (
            f"### Invalid image\n{result.validation_message}\n\n"
            "Please upload a valid greyscale chest CT scan.",
            "",
            None,
        )

    pred_color = "#ff4c4c" if result.prediction == "CANCER" else "#00e676"
    flag_text = (
        "Flag for expert review (high ensemble disagreement)"
        if result.flag_for_review
        else "Confident prediction"
    )
    member_str = " | ".join(
        f"M{i + 1}: {p:.3f}" for i, p in enumerate(result.member_probs)
    )
    bar_pct = int(result.calibrated_prob * 100)

    summary = f"""
<div style="font-family: system-ui, sans-serif; line-height: 1.6;">
  <h2 style="color:{pred_color}; margin-bottom: 8px;">{result.prediction}</h2>
  <div style="background:#1a2d40;border-radius:6px;height:16px;width:100%;max-width:360px;overflow:hidden;margin:8px 0;">
    <div style="background:{pred_color};width:{bar_pct}%;height:100%;"></div>
  </div>
  <p><b>P(cancer) calibrated:</b> {result.calibrated_prob:.4f}</p>
  <p><b>P(cancer) raw:</b> {result.raw_prob:.4f}</p>
  <p><b>Threshold:</b> {result.threshold:.2f} &nbsp;|&nbsp;
     <b>Ensemble variance:</b> {result.uncertainty:.6f}</p>
  <p style="font-size:0.9em;color:#5d8aa8;">Per-member: {member_str}</p>
  <p><b>{flag_text}</b></p>
  <p style="font-size:0.85em;color:#888;">
    Research/demo tool only — not a medical diagnosis.
  </p>
</div>
"""
    details = (
        f"Validation: {result.validation_message}\n"
        f"Grad-CAM agreement: {result.agreement:.3f}"
        if result.agreement is not None
        else f"Validation: {result.validation_message}"
    )
    return summary, details, result.gradcam_image


def predict(
    ct_image: Image.Image | None,
    age: int,
    gender: str,
    smoking: str,
    mc_passes: int,
    threshold: float,
):
    if ct_image is None:
        return (
            "### Upload a CT image first",
            "",
            None,
        )

    try:
        result = predictor.predict(
            image=ct_image,
            age=age,
            gender=gender,
            smoking=smoking,
            mc_passes=mc_passes,
            threshold=threshold,
            include_gradcam=True,
        )
        return _format_result(result)
    except FileNotFoundError as exc:
        return (f"### Model not ready\n\n{exc}", "", None)
    except Exception as exc:
        return (f"### Prediction failed\n\n{exc}", "", None)


def build_ui() -> gr.Blocks:
    test_metrics = {}
    try:
        predictor.load()
        test_metrics = predictor.config.get("test_metrics", {})
        status = (
            f"Model loaded ({len(predictor.ensemble_models)} ensemble members) "
            f"on {predictor.device}"
        )
    except Exception as exc:
        status = f"Waiting for checkpoint files: {exc}"

    with gr.Blocks(
        title="MultiModalOptTab — Lung Cancer Risk",
    ) as demo:
        gr.Markdown(
            """
# MultiModalOptTab — Early Lung Cancer Detection

**EfficientNet-B0 + TabNet + BNN** | 5-model ensemble | MC Dropout | Platt calibration

Upload a chest CT scan and enter patient details to get a cancer-risk estimate with
Grad-CAM++ explainability.
            """
        )
        gr.Markdown(f"**Status:** {status}")

        if test_metrics:
            gr.Markdown(
                f"*Held-out test metrics from training: "
                f"Accuracy {test_metrics.get('acc', 0):.3f} | "
                f"AUC {test_metrics.get('auc_roc', 0):.3f} | "
                f"F1 {test_metrics.get('f1', 0):.3f}*"
            )

        with gr.Row():
            with gr.Column(scale=1):
                ct_image = gr.Image(
                    label="Chest CT scan (greyscale JPG/PNG)",
                    type="pil",
                    image_mode="L",
                )
                age = gr.Slider(18, 90, value=55, step=1, label="Age (years)")
                gender = gr.Dropdown(
                    ["Male", "Female"], value="Male", label="Gender"
                )
                smoking = gr.Dropdown(
                    ["Never Smoked", "Former Smoker", "Current Smoker"],
                    value="Never Smoked",
                    label="Smoking status",
                )
                mc_passes = gr.Slider(
                    5, 50, value=20, step=5, label="MC dropout passes per model"
                )
                threshold = gr.Slider(
                    0.1,
                    0.9,
                    value=float(
                        predictor.best_threshold
                        if predictor._loaded
                        else 0.54
                    ),
                    step=0.01,
                    label="Decision threshold",
                )
                predict_btn = gr.Button("Predict cancer risk", variant="primary")

            with gr.Column(scale=1):
                result_html = gr.Markdown(label="Result")
                result_details = gr.Textbox(label="Details", lines=4)
                gradcam = gr.Image(label="Grad-CAM++ (ensemble average)", type="numpy")

        predict_btn.click(
            fn=predict,
            inputs=[ct_image, age, gender, smoking, mc_passes, threshold],
            outputs=[result_html, result_details, gradcam],
        )

        gr.Markdown(
            """
<p class="disclaimer">
<b>Disclaimer:</b> This is a research and demonstration tool. It is not FDA-approved
and must not be used for clinical diagnosis or treatment decisions.
</p>
            """,
            elem_classes=["disclaimer"],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .disclaimer { font-size: 0.85em; color: #666; margin-top: 12px; }
        """,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=True,
    )
