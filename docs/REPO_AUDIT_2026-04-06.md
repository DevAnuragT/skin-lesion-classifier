# Repo Audit

Date: 2026-04-06

## Goal

Determine what this cloned repository already provides, what is missing locally, and what can be safely claimed in a course progress update.

## Proposal-to-repo match

The project summary aimed to build:

- a multi-class skin disease classifier
- using transfer learning and CNNs
- with preprocessing and model evaluation
- exposed through a Flask web application

The repo broadly matches that scope.

## What already exists in the repo

### Analysis and modelling assets

- `Skin_Disease_Project.ipynb`
- `Notebooks/Business_understanding.ipynb`
- `Notebooks/scraping.ipynb`
- `Notebooks/data_cleaning.ipynb`
- `Notebooks/modelling.ipynb`

These indicate the upstream team already documented business understanding, data collection, cleaning, and model training work.

### Web app

The Flask app exists in `website/main.py` and includes:

- app creation
- model loading
- image upload handling
- file-type validation
- skin detection
- preprocessing
- prediction
- treatment lookup
- rendered output pages

Relevant implementation points:

- `website/main.py:12` creates the Flask app
- `website/main.py:15` loads `skin_disorder_classifier_EfficientNetB2.h5`
- `website/main.py:49` defines `/predict`
- `website/main.py:73` runs model inference
- `website/main.py:98` returns the results page

### Templates and content

- `website/templates/index.html`
- `website/templates/results.html`
- `website/templates/error.html`
- `website/skin_disorder.json`

The treatment JSON contains 8 disease classes, which aligns with the app's class list.

## What is missing or blocked locally

### Real model artifact is missing

`website/skin_disorder_classifier_EfficientNetB2.h5` is not the actual model file in this clone. It is a Git LFS pointer:

- `version https://git-lfs.github.com/spec/v1`
- `size 98750424`

That means the local repo does not currently contain the real trained model needed to run inference.

### Runtime dependencies are not installed in this environment

Local checks showed these packages are missing:

- `tensorflow`
- `keras`
- `numpy`
- `cv2`

Without them, the app cannot start successfully here.

### Training data is incomplete in the clone

`Data/data1-294.csv` exists, but it is a URL listing for scraped images, not a ready local image dataset. The actual training image folders and full ISIC data needed for retraining are not present in this clone.

## Accuracy claim risk

The README says the model achieved 88% accuracy:

- `README.md:88`

But notebook outputs also show lower values in multiple places:

- validation accuracy around `0.6513`
- test accuracy around `0.6025`
- test accuracy around `0.5155`

Conclusion: reported performance is not yet safe to present as a final verified result without rerunning or tracing which experiment produced which number.

## Safe conclusion

This repo gives us a strong upstream baseline and enough material to explain the intended system architecture, but this local clone is not yet a fully reproduced, verified project environment.

The honest current status is:

- baseline selected
- repo audited
- architecture understood
- blockers identified
- local reproduction still pending
