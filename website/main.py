import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
RETRAINED_CANDIDATES = [
    (
        BASE_DIR / "models" / "skin_classifier.keras",
        BASE_DIR / "models" / "class_names.json",
    ),
    (
        BASE_DIR.parent / "artifacts" / "retrained_hf_strict_80" / "skin_classifier.keras",
        BASE_DIR.parent / "artifacts" / "retrained_hf_strict_80" / "class_names.json",
    ),
    (
        BASE_DIR.parent / "artifacts" / "retrained_hf" / "skin_classifier.keras",
        BASE_DIR.parent / "artifacts" / "retrained_hf" / "class_names.json",
    ),
]

VIT_CANDIDATE_DIRS = [
    BASE_DIR / "models",
    BASE_DIR.parent / "artifacts" / "vit_150_tiny224_lowaug",
    BASE_DIR.parent / "artifacts" / "vit_150_tiny160",
    BASE_DIR.parent / "artifacts" / "vit_strict80_tiny160",
    BASE_DIR.parent / "artifacts" / "vit_benchmark_cpu",
]
VIT_ENSEMBLE_METADATA = BASE_DIR / "models" / "vit_ensemble.json"
VIT_PROMOTED_MODEL = BASE_DIR / "models" / "skin_classifier_vit.pt"
VIT_PROMOTED_CLASSES = BASE_DIR / "models" / "class_names_vit.json"

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from PIL import Image, UnidentifiedImageError
import io
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Creating the app
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

LEGACY_CLASSES = [
    "Acne",
    "Basal cell carcinoma",
    "Benign Keratosis-like Lesions (BKL)",
    "Atopic dermatitis(Eczema)",
    "Actinic keratosis(AK)",
    "Melanoma",
    "Psoriasis",
    "Tinea(Ringworm)",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _resolve_path(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute():
        return raw
    return BASE_DIR.parent / raw


def _collect_vit_candidates() -> list[tuple[Path, Path, Path | None, float]]:
    candidates: list[tuple[Path, Path, Path | None, float]] = []
    for candidate_dir in VIT_CANDIDATE_DIRS:
        class_paths = [candidate_dir / "class_names_vit.json", candidate_dir / "class_names.json"]
        for model_name in ("skin_classifier_vit.pt", "best_model.pt"):
            model_path = candidate_dir / model_name
            if not model_path.exists():
                continue
            class_path = next((path for path in class_paths if path.exists()), None)
            if class_path is None:
                continue
            metrics_path = candidate_dir / "metrics.json"
            accuracy = -1.0
            if metrics_path.exists():
                try:
                    accuracy = float(_read_json(metrics_path).get("accuracy", -1.0))
                except Exception:
                    accuracy = -1.0
            candidates.append((model_path, class_path, metrics_path if metrics_path.exists() else None, accuracy))
    candidates.sort(key=lambda item: item[3], reverse=True)
    return candidates


def _load_vit_runtime(model_path: Path, class_path: Path) -> dict[str, Any]:
    import torch
    import timm

    metadata = _read_json(class_path)
    checkpoint = torch.load(model_path, map_location="cpu")

    class_order = metadata["class_order"]
    classes = [metadata["app_labels"][class_key] for class_key in class_order]
    image_size = int(checkpoint.get("image_size", metadata.get("image_size", 224)))
    model_name = checkpoint.get("model_name", metadata.get("model_name", "vit_tiny_patch16_224"))
    state_dict = checkpoint.get("state_dict", checkpoint)

    vit_model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_order),
        img_size=image_size,
    )
    vit_model.load_state_dict(state_dict, strict=True)
    vit_model.eval()

    return {
        "backend": "pytorch_vit",
        "model": vit_model,
        "classes": classes,
        "image_size": image_size,
        "model_path": str(model_path),
        "torch": torch,
    }


def _load_vit_ensemble_runtime(metadata_path: Path) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    members_raw = metadata.get("members", [])
    if not members_raw:
        raise RuntimeError("vit ensemble metadata has no members")

    members: list[dict[str, Any]] = []
    classes: list[str] | None = None
    class_order: list[str] | None = None
    weight_sum = 0.0

    for member_cfg in members_raw:
        model_path = _resolve_path(member_cfg["model_path"])
        class_path = _resolve_path(member_cfg["class_path"])
        weight = float(member_cfg.get("weight", 1.0))

        runtime = _load_vit_runtime(model_path, class_path)
        member_order = _read_json(class_path)["class_order"]
        if class_order is None:
            class_order = member_order
            classes = runtime["classes"]
        elif member_order != class_order:
            raise RuntimeError(f"class order mismatch for ensemble member: {model_path}")

        members.append(
            {
                "model": runtime["model"],
                "image_size": runtime["image_size"],
                "weight": weight,
            }
        )
        weight_sum += weight

    if classes is None or weight_sum <= 0:
        raise RuntimeError("invalid ensemble configuration")

    return {
        "backend": "pytorch_vit_ensemble",
        "model": members[0]["model"],
        "members": members,
        "classes": classes,
        "image_size": max(member["image_size"] for member in members),
        "model_path": str(metadata_path),
        "torch": __import__("torch"),
        "weight_sum": weight_sum,
    }


def _load_keras_runtime() -> dict[str, Any] | None:
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None

    for model_path, class_path in RETRAINED_CANDIDATES:
        if model_path.exists() and class_path.exists():
            metadata = _read_json(class_path)
            class_order = metadata["class_order"]
            classes = [metadata["app_labels"][class_key] for class_key in class_order]
            image_size = metadata.get("image_size", 224)
            return {
                "backend": "keras",
                "model": load_model(model_path),
                "classes": classes,
                "image_size": image_size,
                "model_path": str(model_path),
            }

    legacy_model_path = BASE_DIR / "skin_disorder_classifier_EfficientNetB2.h5"
    if legacy_model_path.exists() and legacy_model_path.stat().st_size > 1_000_000:
        return {
            "backend": "keras",
            "model": load_model(legacy_model_path),
            "classes": LEGACY_CLASSES,
            "image_size": 300,
            "model_path": str(legacy_model_path),
        }
    return None


def load_runtime_model() -> dict[str, Any]:
    if VIT_ENSEMBLE_METADATA.exists():
        try:
            runtime = _load_vit_ensemble_runtime(VIT_ENSEMBLE_METADATA)
            print(f"[runtime] using PyTorch ViT ensemble: {runtime['model_path']}", flush=True)
            return runtime
        except Exception as exc:
            print(f"[runtime] skipped ViT ensemble {VIT_ENSEMBLE_METADATA}: {exc}", flush=True)

    # Prefer the best available ViT artifact if it exists.
    for model_path, class_path, _, _ in _collect_vit_candidates():
        try:
            runtime = _load_vit_runtime(model_path, class_path)
            print(f"[runtime] using PyTorch ViT model: {runtime['model_path']}", flush=True)
            return runtime
        except Exception as exc:
            print(f"[runtime] skipped ViT candidate {model_path}: {exc}", flush=True)

    # Fallback to Keras retrained artifacts or legacy model.
    runtime = _load_keras_runtime()
    if runtime is not None:
        print(f"[runtime] using Keras model: {runtime['model_path']}", flush=True)
        return runtime

    raise RuntimeError("No usable runtime model was found for website inference")


def predict_with_keras(runtime: dict[str, Any], image: Image.Image) -> np.ndarray:
    from tensorflow.keras.preprocessing.image import img_to_array

    img = image.resize((runtime["image_size"], runtime["image_size"]))
    img_array = img_to_array(img)
    batch = np.expand_dims(img_array, axis=0)
    probabilities = runtime["model"].predict(batch, verbose=0)[0]
    return probabilities


def predict_with_vit(runtime: dict[str, Any], image: Image.Image) -> np.ndarray:
    if runtime["backend"] == "pytorch_vit_ensemble":
        aggregate = None
        for member in runtime["members"]:
            resized = image.resize((member["image_size"], member["image_size"]))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            tensor = runtime["torch"].from_numpy(array).permute(2, 0, 1).unsqueeze(0)

            with runtime["torch"].no_grad():
                logits = member["model"](tensor)
                probabilities = runtime["torch"].softmax(logits, dim=1).cpu().numpy()[0]

            weighted = probabilities * float(member["weight"])
            aggregate = weighted if aggregate is None else aggregate + weighted

        return aggregate / float(runtime["weight_sum"])

    resized = image.resize((runtime["image_size"], runtime["image_size"]))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = runtime["torch"].from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    with runtime["torch"].no_grad():
        logits = runtime["model"](tensor)
        probabilities = runtime["torch"].softmax(logits, dim=1).cpu().numpy()[0]
    return probabilities


# Loading the model
RUNTIME = load_runtime_model()
model = RUNTIME["model"]
classes = RUNTIME["classes"]
image_size = RUNTIME["image_size"]

FALLBACK_RUNTIME = None
if RUNTIME["backend"] == "pytorch_vit_ensemble":
    try:
        if VIT_PROMOTED_MODEL.exists() and VIT_PROMOTED_CLASSES.exists():
            candidate = _load_vit_runtime(VIT_PROMOTED_MODEL, VIT_PROMOTED_CLASSES)
            if candidate["classes"] == classes:
                FALLBACK_RUNTIME = candidate
                print("[runtime] enabled low-confidence fallback model", flush=True)
    except Exception as exc:
        print(f"[runtime] skipped fallback model: {exc}", flush=True)

# Loading the json file with the skin disorders
def get_treatment(path):
    with open(path) as f:
        return json.load(f)
treatment_dict = get_treatment(BASE_DIR / "skin_disorder.json")

TREATMENT_ALIASES = {
    "Actinic keratosis": "Actinic keratosis(AK)",
    "Actinic keratosis(AK)": "Actinic keratosis(AK)",
    "Tinea(Ringworm)": "Tinea (Ringworm)",
    "Tinea (Ringworm)": "Tinea (Ringworm)",
}


def get_treatments_for_prediction(predicted_label):
    canonical_label = TREATMENT_ALIASES.get(predicted_label, predicted_label)
    return treatment_dict.get(canonical_label, [])


def confidence_band(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.55:
        return "Moderate"
    if probability >= 0.4:
        return "Low"
    return "Very low"

# function to check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# function to detect skin color
def is_skin(img):
    # convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # create a binary mask of skin color pixels
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # count the number of skin color pixels
    skin_pixels = np.sum(mask > 0)
    # calculate the percentage of skin color pixels in the image
    skin_percent = skin_pixels / (img.shape[0] * img.shape[1]) * 100
    # return True if skin percentage is above a threshold, else False
    return skin_percent > 5

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['file']

    # check if the file is an image
    if not file or not allowed_file(file.filename):
        return render_template('error.html', error='Only image files are allowed')

    # Open the image using PIL
    try:
        image = Image.open(file).convert("RGB")
    except UnidentifiedImageError:
        return render_template(
            'error.html',
            error='The uploaded file is not a valid readable image. Please try another JPG or PNG file.'
        )
    except Exception:
        return render_template(
            'error.html',
            error='The uploaded image could not be opened. Please try another image file.'
        )

    # Make prediction using the active backend.
    if RUNTIME["backend"] in {"pytorch_vit", "pytorch_vit_ensemble"}:
        pred = predict_with_vit(RUNTIME, image)
    else:
        pred = predict_with_keras(RUNTIME, image)

    # For harder web images, ensemble confidence can be under-calibrated.
    # If confidence is low, compare against the promoted single ViT model.
    if RUNTIME["backend"] == "pytorch_vit_ensemble" and FALLBACK_RUNTIME is not None:
        ensemble_conf = float(np.max(pred))
        if ensemble_conf < 0.45:
            fallback_pred = predict_with_vit(FALLBACK_RUNTIME, image)
            if float(np.max(fallback_pred)) > ensemble_conf:
                pred = fallback_pred

    class_idx = int(np.argmax(pred))
    sorted_indices = np.argsort(pred)[::-1]

    # Predicted class
    pred_class = classes[class_idx]
    second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else class_idx
    second_prob = float(pred[second_idx])

    # Probability of prediction
    prob = float(pred[class_idx])

    # These web/phone samples are often less calibrated under ViT softmax.
    # Use a lower threshold and combine it with a soft skin-content gate.
    threshold = 0.32 if RUNTIME["backend"] in {"pytorch_vit", "pytorch_vit_ensemble"} else 0.6
    skin_detected = is_skin(np.array(image))

    if prob < threshold:
        return render_template('error.html', error='Inconclusive result.\
                                                    Please consult a healthcare professional for an accurate diagnosis')

    if not skin_detected and prob < 0.35:
        return render_template('error.html', error='The uploaded image could not be processed.\
                                                    Please ensure that the image contains a clear skin lesion and try again.')

    # Additional ambiguity check for near-tie outcomes.
    if prob < 0.55 and (prob - second_prob) < 0.12:
        return render_template('error.html', error='Inconclusive result.\
                                                    The top predictions are too close for a reliable result.')

    # Treatment options
    treatments = get_treatments_for_prediction(pred_class)

    # Render the results page with the prediction
    return render_template(
        'results.html',
        prediction=pred_class,
        probability=prob,
        confidence_percentage=round(float(prob) * 100, 1),
        confidence_band=confidence_band(prob),
        top_predictions=[
            {
                "label": classes[int(idx)],
                "probability": round(float(pred[int(idx)]) * 100, 1),
            }
            for idx in sorted_indices[:3]
        ],
        treatments=treatments,
    )

# Run the application   
if __name__ == '__main__':
    app.run(debug=True)
