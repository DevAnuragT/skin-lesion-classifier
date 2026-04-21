#!/usr/bin/env python3
"""Optimize weighted ViT ensemble on validation split and report test accuracy."""

from __future__ import annotations

import argparse
import itertools
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
    parser.add_argument(
        "--candidates-json",
        default="website/models/vit_ensemble_candidates.json",
        help="JSON list with candidate model_path and class_path entries",
    )
    parser.add_argument("--val-split", default="datasets/hf_hybrid_150/splits/validation")
    parser.add_argument("--test-split", default="datasets/hf_hybrid_150/splits/test")
    parser.add_argument("--output-config", default="website/models/vit_ensemble.json")
    parser.add_argument("--report-json", default="artifacts/ensemble_optimization_report.json")
    return parser.parse_args()


def resolve_path(repo_root: Path, text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return repo_root / path


def load_models(repo_root: Path, candidates: list[dict]) -> tuple[list[dict], list[str]]:
    members = []
    class_order = None
    for item in candidates:
        model_path = resolve_path(repo_root, item["model_path"])
        class_path = resolve_path(repo_root, item["class_path"])
        checkpoint = torch.load(model_path, map_location="cpu")
        metadata = json.loads(class_path.read_text(encoding="utf-8"))
        this_order = metadata["class_order"]

        if class_order is None:
            class_order = this_order
        elif this_order != class_order:
            raise RuntimeError(f"class order mismatch: {model_path}")

        model = timm.create_model(
            checkpoint.get("model_name", metadata.get("model_name", "vit_tiny_patch16_224")),
            pretrained=False,
            num_classes=len(this_order),
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

        members.append(
            {
                "name": item["name"],
                "model_path": str(model_path),
                "class_path": str(class_path),
                "model": model,
                "tf": tf,
            }
        )

    assert class_order is not None
    return members, class_order


def collect_items(split_dir: Path, class_order: list[str]) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for index, class_key in enumerate(class_order):
        class_dir = split_dir / class_key
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                items.append((image_path, index))
    return items


def compute_prob_cache(members: list[dict], items: list[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
    probs = {member["name"]: [] for member in members}
    y_true = []

    for image_path, target in items:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        y_true.append(target)
        for member in members:
            tensor = member["tf"](image).unsqueeze(0)
            with torch.no_grad():
                logits = member["model"](tensor)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs[member["name"]].append(p)

    stacked = np.stack([np.stack(probs[member["name"]], axis=0) for member in members], axis=0)
    return stacked, np.array(y_true)


def ensemble_accuracy(prob_stack: np.ndarray, y_true: np.ndarray, weights: np.ndarray) -> float:
    weighted = np.tensordot(weights, prob_stack, axes=(0, 0)) / np.sum(weights)
    pred = np.argmax(weighted, axis=1)
    return float(np.mean(pred == y_true))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    candidates_path = resolve_path(repo_root, args.candidates_json)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    members, class_order = load_models(repo_root, candidates)
    val_items = collect_items(resolve_path(repo_root, args.val_split), class_order)
    test_items = collect_items(resolve_path(repo_root, args.test_split), class_order)

    val_probs, val_y = compute_prob_cache(members, val_items)
    test_probs, test_y = compute_prob_cache(members, test_items)

    grid = [0.25, 0.4, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0]

    best = {
        "val_accuracy": -1.0,
        "weights": None,
    }

    for combo in itertools.product(grid, repeat=len(members)):
        weights = np.array(combo, dtype=np.float32)
        val_acc = ensemble_accuracy(val_probs, val_y, weights)
        if val_acc > best["val_accuracy"]:
            best["val_accuracy"] = val_acc
            best["weights"] = weights.copy()

    assert best["weights"] is not None
    best_test_acc = ensemble_accuracy(test_probs, test_y, best["weights"])

    output_members = []
    for member, weight in zip(members, best["weights"].tolist()):
        output_members.append(
            {
                "model_path": str(Path(member["model_path"]).relative_to(repo_root)),
                "class_path": str(Path(member["class_path"]).relative_to(repo_root)),
                "weight": float(weight),
            }
        )

    output_config = {
        "name": "vit_weighted_ensemble_optimized_on_validation",
        "selection_basis": "validation grid search over candidate model weights",
        "members": output_members,
    }

    output_config_path = resolve_path(repo_root, args.output_config)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(output_config, indent=2), encoding="utf-8")

    report = {
        "candidate_count": len(members),
        "val_accuracy": best["val_accuracy"],
        "test_accuracy": best_test_acc,
        "weights": {member["name"]: float(weight) for member, weight in zip(members, best["weights"])},
        "output_config": str(output_config_path),
    }

    report_path = resolve_path(repo_root, args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
