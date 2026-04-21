# Local Setup Next Steps

This file turns the repo audit into the next reproducible setup task list.

## 1. Create a Python environment

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r website/requirements.txt
```

## 2. Install Git LFS and fetch the real model artifact

The current `website/skin_disorder_classifier_EfficientNetB2.h5` file in this clone is only a Git LFS pointer, not the actual model.

Example sequence:

```bash
git lfs install
git lfs pull
```

After that, confirm the file is no longer a small text pointer and has a realistic binary size.

## 3. Run the Flask app from the `website/` directory

Example:

```bash
cd website
python3 main.py
```

## 4. Validate the minimum working flow

- App starts without import errors.
- Home page loads.
- Valid image upload reaches prediction flow.
- Invalid file upload shows the error page.
- Low-confidence prediction triggers the inconclusive-result path.

## 5. Validate performance claims separately

Do not treat the README's 88% figure as final until the notebook outputs and saved model version are traced to the same training run.

## Current blocker summary

- Missing runtime packages in the current environment
- Missing real LFS model artifact in the current clone
- Incomplete local training data for retraining
- Inconsistent reported metrics across notebook and README
