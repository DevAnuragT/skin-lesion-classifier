# Skin Lesion Classifier

Skin Lesion Classifier is a deep learning based web application for preliminary skin condition recognition from uploaded images. The project combines an image classification pipeline with a Flask interface so a user can upload a skin image and receive a predicted class with treatment-oriented reference information.

## Supported classes

- Acne
- Basal Cell Carcinoma
- Benign Keratosis-like Lesions
- Atopic Dermatitis (Eczema)
- Actinic Keratosis
- Melanoma
- Psoriasis
- Tinea (Ringworm)

## Repository contents

- `website/`
  Flask application, templates, and tracked runtime model files
- `scripts/`
  Dataset-building and retraining pipelines
- `Notebooks/`
  exploratory notebooks for scraping, cleaning, and modelling
- `Data/`
  source metadata used during dataset preparation

## Current model status

- The app is configured to prefer the tracked retrained model in `website/models/`.
- A cleaner retraining run is also preserved in `artifacts/` locally for experimentation.
- The project should still be treated as a prototype, not a medical diagnosis system.

## Run locally

From the repository root:

```bash
bash run_local.sh
```

Or manually:

```bash
source .venv/bin/activate
cd website
python main.py
```

Then open:

```text
http://127.0.0.1:5000
```

More run details are in [docs/RUN_LOCAL.md](/home/anurag/Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis/docs/RUN_LOCAL.md).

## Training pipeline

The repository includes reproducible scripts for dataset construction and retraining:

- `scripts/build_hf_hybrid_dataset.py`
- `scripts/train_model.py`

These scripts were used to create balanced 8-class datasets and evaluate replacement model artifacts.
