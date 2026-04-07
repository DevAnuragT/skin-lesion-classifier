# Run Locally

This project is configured to run from the local virtual environment and uses the tracked retrained model in `website/models/` by default.

## Fastest option

From the repo root:

```bash
bash run_local.sh
```

## 1. Activate the environment

From the repo root:

```bash
source .venv/bin/activate
```

## 2. Start the Flask app

Run from the `website/` directory:

```bash
cd website
python main.py
```

Expected behavior:

- TensorFlow will print some CPU and CUDA-related logs.
- Flask should start in debug mode.
- Open the local URL shown by Flask in your browser.

## 3. What to test

- Home page loads
- Uploading a non-image file shows an error
- Uploading a skin image goes through prediction
- Low-confidence predictions show the inconclusive message

## 4. Notes

- The app prefers `website/models/skin_classifier.keras` and falls back to the legacy `.h5` file only if the retrained model is missing.
- TensorFlow will still print CPU and CUDA related logs on startup.
- In this Codex sandbox, opening a listening port is blocked, so validation here was done with Flask's test client instead of a live browser.

## 5. If the venv is missing later

Recreate it with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r website/requirements.txt
```

## 6. Model files

Tracked runtime model files:

- `website/models/skin_classifier.keras`
- `website/models/class_names.json`
