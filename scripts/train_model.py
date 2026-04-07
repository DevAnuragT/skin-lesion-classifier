#!/usr/bin/env python3
"""Train and export a replacement skin lesion classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CLASS_ORDER = ["acne", "bcc", "bkl", "eczema", "ak", "melanoma", "psoriasis", "tinea"]
APP_LABELS = {
    "acne": "Acne",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign Keratosis-like Lesions (BKL)",
    "eczema": "Atopic dermatitis(Eczema)",
    "ak": "Actinic keratosis(AK)",
    "melanoma": "Melanoma",
    "psoriasis": "Psoriasis",
    "tinea": "Tinea(Ringworm)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/derma_auto/splits")
    parser.add_argument("--output-dir", default="artifacts/retrained")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--initial-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    return parser.parse_args()


def load_dataset(split_dir: Path, image_size: int, batch_size: int, seed: int, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_ORDER,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
        seed=seed,
    )


def prepare_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    if training:
        dataset = dataset.shuffle(512)
    return dataset.prefetch(autotune)


def build_model(image_size: int) -> tuple[keras.Model, keras.Model]:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = augmentation(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(CLASS_ORDER), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: keras.Model,
    base_model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, list[float]]]:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_accuracy", save_best_only=True),
    ]

    compile_model(model, args.learning_rate)
    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        callbacks=callbacks,
        verbose=2,
    ).history

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, args.fine_tune_learning_rate)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs + args.fine_tune_epochs,
        initial_epoch=len(initial_history["loss"]),
        callbacks=callbacks,
        verbose=2,
    ).history

    return [initial_history, fine_tune_history]


def dataset_arrays(dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    images_list = []
    labels_list = []
    for batch_images, batch_labels in dataset:
        images_list.append(batch_images.numpy())
        labels_list.append(np.argmax(batch_labels.numpy(), axis=1))
    return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)


def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset, output_dir: Path) -> dict[str, object]:
    test_images, y_true = dataset_arrays(test_ds)
    probabilities = model.predict(test_images, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_ORDER)).numpy()
    accuracy = float(np.mean(y_true == y_pred))

    per_class = {}
    for index, class_key in enumerate(CLASS_ORDER):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[class_key] = {
            "app_label": APP_LABELS[class_key],
            "precision": precision,
            "recall": recall,
            "support": int(confusion[index, :].sum()),
        }

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_ORDER])
        for class_key, row in zip(CLASS_ORDER, confusion):
            writer.writerow([class_key, *map(int, row.tolist())])

    prediction_rows = []
    for index, class_key in enumerate(CLASS_ORDER):
        prediction_rows.append(
            {
                "class_key": class_key,
                "app_label": APP_LABELS[class_key],
                "predicted_count": int((y_pred == index).sum()),
                "true_count": int((y_true == index).sum()),
            }
        )

    metrics = {
        "accuracy": accuracy,
        "per_class": per_class,
        "prediction_counts": prediction_rows,
    }
    return metrics


def write_outputs(model: keras.Model, output_dir: Path, histories: list[dict[str, list[float]]], metrics: dict[str, object], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "skin_classifier.keras")
    model.save(output_dir / "skin_classifier.h5")
    (output_dir / "class_names.json").write_text(
        json.dumps(
            {
                "class_order": CLASS_ORDER,
                "app_labels": APP_LABELS,
                "image_size": args.image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = prepare_dataset(
        load_dataset(Path(args.data_root) / "train", args.image_size, args.batch_size, args.seed, shuffle=True),
        training=True,
    )
    val_ds = prepare_dataset(
        load_dataset(Path(args.data_root) / "validation", args.image_size, args.batch_size, args.seed, shuffle=False),
        training=False,
    )
    test_ds = prepare_dataset(
        load_dataset(Path(args.data_root) / "test", args.image_size, args.batch_size, args.seed, shuffle=False),
        training=False,
    )

    model, base_model = build_model(args.image_size)
    histories = train_model(model, base_model, train_ds, val_ds, output_dir, args)
    metrics = evaluate_model(model, test_ds, output_dir)
    write_outputs(model, output_dir, histories, metrics, args)

    print("[training] complete")
    print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
