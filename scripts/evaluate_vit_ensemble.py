#!/usr/bin/env python3
"""Evaluate configured ViT ensemble on a folder-based split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-config", default="website/models/vit_ensemble.json")
    parser.add_argument("--split-dir", default="datasets/hf_hybrid_150/splits/test")
    parser.add_argument("--output-json", default="website/models/vit_ensemble_metrics.json")
    parser.add_argument("--output-confusion-csv", default="website/models/vit_ensemble_confusion.csv")
    return parser.parse_args()


def resolve_path(repo_root: Path, text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return repo_root / path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.ensemble_config)
    split_dir = resolve_path(repo_root, args.split_dir)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    members = config.get("members", [])
    if not members:
        raise RuntimeError("ensemble config has no members")

    loaded = []
    class_order = None

    for member in members:
        model_path = resolve_path(repo_root, member["model_path"])
        class_path = resolve_path(repo_root, member["class_path"])
        weight = float(member.get("weight", 1.0))

        checkpoint = torch.load(model_path, map_location="cpu")
        metadata = json.loads(class_path.read_text(encoding="utf-8"))

        current_order = metadata["class_order"]
        if class_order is None:
            class_order = current_order
        elif current_order != class_order:
            raise RuntimeError(f"class order mismatch: {model_path}")

        model = timm.create_model(
            checkpoint.get("model_name", metadata.get("model_name", "vit_tiny_patch16_224")),
            pretrained=False,
            num_classes=len(current_order),
            img_size=int(checkpoint.get("image_size", metadata.get("image_size", 224))),
        )
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        model.eval()

        image_size = int(checkpoint.get("image_size", metadata.get("image_size", 224)))
        tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        loaded.append({"model": model, "tf": tf, "weight": weight})

    y_true = []
    y_pred = []

    for class_index, class_key in enumerate(class_order):
        class_dir = split_dir / class_key
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")

            weighted_probs = None
            total_weight = 0.0
            for member in loaded:
                tensor = member["tf"](image).unsqueeze(0)
                with torch.no_grad():
                    logits = member["model"](tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                weighted = probs * member["weight"]
                weighted_probs = weighted if weighted_probs is None else weighted_probs + weighted
                total_weight += member["weight"]

            probs = weighted_probs / total_weight
            prediction = int(np.argmax(probs))

            y_true.append(class_index)
            y_pred.append(prediction)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    confusion = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    for t, p in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
        confusion[t, p] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_key in enumerate(class_order):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[class_key] = {
            "precision": precision,
            "recall": recall,
            "support": int(confusion[index, :].sum()),
        }

    macro_precision = float(np.mean([per_class[class_key]["precision"] for class_key in class_order]))
    macro_recall = float(np.mean([per_class[class_key]["recall"] for class_key in class_order]))

    prediction_counts = []
    for index, class_key in enumerate(class_order):
        prediction_counts.append(
            {
                "class_key": class_key,
                "predicted_count": int((y_pred_arr == index).sum()),
                "true_count": int((y_true_arr == index).sum()),
            }
        )

    report = {
        "ensemble_config": str(config_path),
        "split_dir": str(split_dir),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "prediction_counts": prediction_counts,
        "class_order": class_order,
        "samples": len(y_true),
        "members": members,
    }

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    confusion_path = resolve_path(repo_root, args.output_confusion_csv)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8") as handle:
        handle.write("true/pred," + ",".join(class_order) + "\n")
        for class_key, row in zip(class_order, confusion):
            handle.write(class_key + "," + ",".join(str(int(value)) for value in row.tolist()) + "\n")

    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
