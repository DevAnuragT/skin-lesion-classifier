#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
APP_DIR="$ROOT_DIR/website"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing virtual environment at $VENV_PATH"
  echo "Create it first with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r website/requirements.txt"
  exit 1
fi

if [[ ! -f "$APP_DIR/skin_disorder_classifier_EfficientNetB2.h5" ]]; then
  echo "Missing model file at $APP_DIR/skin_disorder_classifier_EfficientNetB2.h5"
  exit 1
fi

if [[ "$(wc -c < "$APP_DIR/skin_disorder_classifier_EfficientNetB2.h5")" -lt 1000000 ]]; then
  echo "Model file looks too small and may still be a Git LFS pointer:"
  echo "  $APP_DIR/skin_disorder_classifier_EfficientNetB2.h5"
  exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$APP_DIR"
exec python main.py
