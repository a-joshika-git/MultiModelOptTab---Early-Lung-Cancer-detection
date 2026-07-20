---
title: MultiModalOptTab
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# MultiModalOptTab — Lung Cancer Risk Demo

Public Gradio demo for the MultiModalOptTab ensemble model.

## Checkpoint files required

Upload these files from notebook **Cell 18** (`/content/drive/MyDrive/lung_cancer/checkpoint/`) into the `checkpoint/` folder:

- `ensemble_member_1.pth` … `ensemble_member_5.pth`
- `config.joblib`
- `qt.joblib`, `le_gender.joblib`, `oe.joblib`, `platt.joblib`
- `ct_ref_profile.joblib` (optional, improves CT validation)

## Disclaimer

Research/demo tool only — not for clinical use.
