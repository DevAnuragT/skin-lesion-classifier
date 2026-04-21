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

For the ViT workflow, use a single orchestrator command from repository root:

```bash
source .venv/bin/activate
python scripts/run_vit_pipeline.py --mode full --skip-existing
```

What this runs automatically:

- selected ViT profiles (unless skipped by existing metrics)
- best-checkpoint promotion into `website/models/`
- metrics summary and docs plot refresh in `docs/plots/`
- website runtime validation for selected backend/model

Ensemble inference support:

- If `website/models/vit_ensemble.json` exists, the Flask runtime will load a weighted ViT ensemble first.
- If ensemble loading fails, runtime safely falls back to single-model ViT / Keras logic.

Accuracy-oriented methods now implemented in `scripts/train_vit.py`:

- augmentation levels (`--aug-level`: `light|medium|strong`)
- optional mixup/cutmix (`--mixup-alpha`, `--cutmix-alpha`, `--mixup-prob`)
- random erasing (`--random-erasing`)
- ViT stochastic depth control (`--drop-path-rate`)
- exponential moving average model tracking (`--ema-decay`)
- two-stage transfer learning (`--freeze-backbone-epochs`)
- optional focal loss (`--loss-type focal`, `--focal-gamma`)
- hard-class sampling (`--class-sampling hard`, `--hard-class-boost`)
- test-time horizontal flip averaging (`--tta-flip`)

To run only the new stronger profile:

```bash
source .venv/bin/activate
python scripts/run_vit_pipeline.py --mode train --profiles hf150_tiny160_v2_strongreg
python scripts/run_vit_pipeline.py --mode promote
python scripts/run_vit_pipeline.py --mode report
python scripts/run_vit_pipeline.py --mode validate
```

These scripts were used to create balanced 8-class datasets and evaluate replacement model artifacts.
