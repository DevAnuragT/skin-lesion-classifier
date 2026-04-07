import os
from pathlib import Path

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

if not any(model_path.exists() and class_path.exists() for model_path, class_path in RETRAINED_CANDIDATES):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import json
from PIL import Image, UnidentifiedImageError
import io
import cv2

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


def load_runtime_model():
    for model_path, class_path in RETRAINED_CANDIDATES:
        if model_path.exists() and class_path.exists():
            with class_path.open() as handle:
                metadata = json.load(handle)
            class_order = metadata["class_order"]
            classes = [metadata["app_labels"][class_key] for class_key in class_order]
            image_size = metadata.get("image_size", 224)
            return load_model(model_path), classes, image_size

    legacy_model_path = BASE_DIR / "skin_disorder_classifier_EfficientNetB2.h5"
    return load_model(legacy_model_path), LEGACY_CLASSES, 300


# Loading the model
model, classes, image_size = load_runtime_model()

# Loading the json file with the skin disorders
def get_treatment(path):
    with open(path) as f:
        return json.load(f)
treatment_dict = get_treatment(BASE_DIR / "skin_disorder.json")

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

    # check if the image contains human skin
    if not is_skin(np.array(image)):
        return render_template('error.html', error='The uploaded image could not be processed.\
                                                    Please ensure that the image contains skin and try again.')

    # Preprocess the image
    img = image.resize((image_size, image_size))
    img_array = img_to_array(img)
    # The saved model already includes an input Rescaling layer.
    # Keep raw pixel values here to avoid normalizing twice.
    img = img_array
    image = np.expand_dims(img, axis=0)

    # Make prediction
    pred = model.predict(image)
    class_idx = np.argmax(pred)

    # Predicted class
    pred_class = classes[class_idx]

    # Probability of prediction
    prob = pred[0][class_idx]

    # Set probability threshold
    threshold = 0.6

    # Check if probability is above threshold
    if prob < threshold:
        return render_template('error.html', error='Inconclusive result.\
                                                    Please consult a healthcare professional for an accurate diagnosis')

    # Treatment options
    treatments = treatment_dict.get(pred_class, [])

    # Render the results page with the prediction
    return render_template(
        'results.html',
        prediction=pred_class,
        probability=prob,
        confidence_percentage=round(float(prob) * 100, 1),
        treatments=treatments,
    )

# Run the application   
if __name__ == '__main__':
    app.run(debug=True)
