#!/usr/bin/env python3
"""Promote the best completed ViT artifact into website/models."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Candidate:
    directory: Path
    model_path: Path
    class_names_path: Path
    metrics_path: Path
    accuracy: float
    macro_precision: float
    macro_recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--prefix", default="vit_")
    parser.add_argument("--website-models-dir", default="website/models")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_candidates(artifacts_root: Path, prefix: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not artifacts_root.exists():
        return candidates

    for directory in sorted(artifacts_root.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefix):
            continue

        model_path = directory / "best_model.pt"
        class_names_path = directory / "class_names.json"
        metrics_path = directory / "metrics.json"

        if not (model_path.exists() and class_names_path.exists() and metrics_path.exists()):
            continue

        metrics = load_json(metrics_path)
        candidates.append(
            Candidate(
                directory=directory,
                model_path=model_path,
                class_names_path=class_names_path,
                metrics_path=metrics_path,
                accuracy=float(metrics.get("accuracy", -1.0)),
                macro_precision=float(metrics.get("macro_precision", -1.0)),
                macro_recall=float(metrics.get("macro_recall", -1.0)),
            )
        )

    # Sort by accuracy, then macro precision, then macro recall.
    candidates.sort(key=lambda c: (c.accuracy, c.macro_precision, c.macro_recall), reverse=True)
    return candidates


def promote(candidate: Candidate, website_models_dir: Path) -> None:
    website_models_dir.mkdir(parents=True, exist_ok=True)

    promoted_model_path = website_models_dir / "skin_classifier_vit.pt"
    promoted_class_path = website_models_dir / "class_names_vit.json"
    promoted_metrics_path = website_models_dir / "metrics.json"
    promoted_history_path = website_models_dir / "history.json"
    promoted_confusion_path = website_models_dir / "confusion_matrix.csv"
    metadata_path = website_models_dir / "vit_runtime_selection.json"

    shutil.copy2(candidate.model_path, promoted_model_path)
    shutil.copy2(candidate.class_names_path, promoted_class_path)
    shutil.copy2(candidate.metrics_path, promoted_metrics_path)

    source_history_path = candidate.directory / "history.json"
    source_confusion_path = candidate.directory / "confusion_matrix.csv"
    if source_history_path.exists():
        shutil.copy2(source_history_path, promoted_history_path)
    if source_confusion_path.exists():
        shutil.copy2(source_confusion_path, promoted_confusion_path)

    metadata = {
        "source_artifact": str(candidate.directory),
        "source_model": str(candidate.model_path),
        "source_class_names": str(candidate.class_names_path),
        "metrics": {
            "accuracy": candidate.accuracy,
            "macro_precision": candidate.macro_precision,
            "macro_recall": candidate.macro_recall,
        },
        "promoted_model": str(promoted_model_path),
        "promoted_class_names": str(promoted_class_path),
        "promoted_metrics": str(promoted_metrics_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    website_models_dir = Path(args.website_models_dir)

    candidates = find_candidates(artifacts_root, args.prefix)
    if not candidates:
        print("[promote] no completed ViT candidates found")
        return 1

    best = candidates[0]
    promote(best, website_models_dir)

    print("[promote] success")
    print(json.dumps(
        {
            "source": str(best.directory),
            "accuracy": best.accuracy,
            "macro_precision": best.macro_precision,
            "macro_recall": best.macro_recall,
            "promoted_to": str(website_models_dir / "skin_classifier_vit.pt"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
