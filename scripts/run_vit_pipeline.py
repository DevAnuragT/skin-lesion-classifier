#!/usr/bin/env python3
"""Run end-to-end ViT training automation: train, promote, report, validate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrainProfile:
    name: str
    data_root: str
    output_dir: str
    model_name: str
    image_size: int
    batch_size: int
    epochs: int
    patience: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    aug_level: str
    mixup_alpha: float
    cutmix_alpha: float
    mixup_prob: float
    random_erasing: float
    drop_path_rate: float
    ema_decay: float
    ema_start_epoch: int
    freeze_backbone_epochs: int
    loss_type: str
    focal_gamma: float
    class_sampling: str
    hard_class_boost: float
    tta_flip: bool
    device: str


DEFAULT_PROFILES = {
    "strict80_tiny160": TrainProfile(
        name="strict80_tiny160",
        data_root="datasets/hf_hybrid_strict_80/splits",
        output_dir="artifacts/vit_strict80_tiny160",
        model_name="vit_tiny_patch16_224",
        image_size=160,
        batch_size=8,
        epochs=18,
        patience=6,
        num_workers=2,
        learning_rate=3e-5,
        weight_decay=0.05,
        label_smoothing=0.1,
        aug_level="medium",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        random_erasing=0.0,
        drop_path_rate=0.1,
        ema_decay=0.0,
        ema_start_epoch=3,
        freeze_backbone_epochs=0,
        loss_type="ce",
        focal_gamma=2.0,
        class_sampling="uniform",
        hard_class_boost=1.0,
        tta_flip=False,
        device="cpu",
    ),
    "hf150_tiny160": TrainProfile(
        name="hf150_tiny160",
        data_root="datasets/hf_hybrid_150/splits",
        output_dir="artifacts/vit_150_tiny160",
        model_name="vit_tiny_patch16_224",
        image_size=160,
        batch_size=8,
        epochs=20,
        patience=7,
        num_workers=2,
        learning_rate=3e-5,
        weight_decay=0.05,
        label_smoothing=0.1,
        aug_level="medium",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        random_erasing=0.0,
        drop_path_rate=0.1,
        ema_decay=0.0,
        ema_start_epoch=3,
        freeze_backbone_epochs=0,
        loss_type="ce",
        focal_gamma=2.0,
        class_sampling="uniform",
        hard_class_boost=1.0,
        tta_flip=False,
        device="cpu",
    ),
    "hf150_tiny224_lowaug": TrainProfile(
        name="hf150_tiny224_lowaug",
        data_root="datasets/hf_hybrid_150/splits",
        output_dir="artifacts/vit_150_tiny224_lowaug",
        model_name="vit_tiny_patch16_224",
        image_size=224,
        batch_size=6,
        epochs=14,
        patience=5,
        num_workers=2,
        learning_rate=2e-5,
        weight_decay=0.03,
        label_smoothing=0.05,
        aug_level="light",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        random_erasing=0.0,
        drop_path_rate=0.08,
        ema_decay=0.0,
        ema_start_epoch=3,
        freeze_backbone_epochs=0,
        loss_type="ce",
        focal_gamma=2.0,
        class_sampling="uniform",
        hard_class_boost=1.0,
        tta_flip=False,
        device="cpu",
    ),
    "hf150_tiny160_v2_strongreg": TrainProfile(
        name="hf150_tiny160_v2_strongreg",
        data_root="datasets/hf_hybrid_150/splits",
        output_dir="artifacts/vit_150_tiny160_v2",
        model_name="vit_tiny_patch16_224",
        image_size=160,
        batch_size=8,
        epochs=26,
        patience=8,
        num_workers=2,
        learning_rate=4e-5,
        weight_decay=0.06,
        label_smoothing=0.1,
        aug_level="strong",
        mixup_alpha=0.2,
        cutmix_alpha=0.5,
        mixup_prob=0.7,
        random_erasing=0.2,
        drop_path_rate=0.15,
        ema_decay=0.9996,
        ema_start_epoch=4,
        freeze_backbone_epochs=0,
        loss_type="ce",
        focal_gamma=2.0,
        class_sampling="uniform",
        hard_class_boost=1.0,
        tta_flip=True,
        device="cpu",
    ),
    "hf150_tiny160_v3_twostage": TrainProfile(
        name="hf150_tiny160_v3_twostage",
        data_root="datasets/hf_hybrid_150/splits",
        output_dir="artifacts/vit_150_tiny160_v3_twostage",
        model_name="vit_tiny_patch16_224",
        image_size=160,
        batch_size=8,
        epochs=20,
        patience=7,
        num_workers=2,
        learning_rate=3e-5,
        weight_decay=0.05,
        label_smoothing=0.08,
        aug_level="medium",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        random_erasing=0.1,
        drop_path_rate=0.12,
        ema_decay=0.999,
        ema_start_epoch=4,
        freeze_backbone_epochs=2,
        loss_type="ce",
        focal_gamma=2.0,
        class_sampling="uniform",
        hard_class_boost=1.0,
        tta_flip=True,
        device="cpu",
    ),
    "hf150_tiny160_v6_hardfocus": TrainProfile(
        name="hf150_tiny160_v6_hardfocus",
        data_root="datasets/hf_hybrid_150/splits",
        output_dir="artifacts/vit_150_tiny160_v6_hardfocus",
        model_name="vit_tiny_patch16_224",
        image_size=160,
        batch_size=8,
        epochs=24,
        patience=8,
        num_workers=2,
        learning_rate=3e-5,
        weight_decay=0.05,
        label_smoothing=0.06,
        aug_level="medium",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        random_erasing=0.08,
        drop_path_rate=0.1,
        ema_decay=0.999,
        ema_start_epoch=4,
        freeze_backbone_epochs=0,
        loss_type="focal",
        focal_gamma=1.5,
        class_sampling="hard",
        hard_class_boost=1.9,
        tta_flip=True,
        device="cpu",
    ),
}

DEFAULT_SELECTED_PROFILES = ["strict80_tiny160", "hf150_tiny160", "hf150_tiny224_lowaug"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("full", "train", "promote", "report", "validate"),
        default="full",
        help="full = train + promote + report + validate",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_SELECTED_PROFILES),
        help="Comma separated training profiles to run in train/full mode",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip train profile when metrics.json already exists")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def selected_profiles(profile_csv: str) -> list[TrainProfile]:
    names = [name.strip() for name in profile_csv.split(",") if name.strip()]
    profiles = []
    for name in names:
        if name not in DEFAULT_PROFILES:
            raise ValueError(f"unknown profile: {name}")
        profiles.append(DEFAULT_PROFILES[name])
    return profiles


def train_profiles(repo_root: Path, python_exe: str, profiles: Iterable[TrainProfile], skip_existing: bool) -> None:
    for profile in profiles:
        metrics_path = repo_root / profile.output_dir / "metrics.json"
        if skip_existing and metrics_path.exists():
            print(f"[skip] {profile.name} already has metrics: {metrics_path}", flush=True)
            continue

        cmd = [
            python_exe,
            "scripts/train_vit.py",
            "--data-root",
            profile.data_root,
            "--output-dir",
            profile.output_dir,
            "--model-name",
            profile.model_name,
            "--image-size",
            str(profile.image_size),
            "--batch-size",
            str(profile.batch_size),
            "--epochs",
            str(profile.epochs),
            "--patience",
            str(profile.patience),
            "--num-workers",
            str(profile.num_workers),
            "--learning-rate",
            str(profile.learning_rate),
            "--weight-decay",
            str(profile.weight_decay),
            "--label-smoothing",
            str(profile.label_smoothing),
            "--aug-level",
            profile.aug_level,
            "--mixup-alpha",
            str(profile.mixup_alpha),
            "--cutmix-alpha",
            str(profile.cutmix_alpha),
            "--mixup-prob",
            str(profile.mixup_prob),
            "--random-erasing",
            str(profile.random_erasing),
            "--drop-path-rate",
            str(profile.drop_path_rate),
            "--ema-decay",
            str(profile.ema_decay),
            "--ema-start-epoch",
            str(profile.ema_start_epoch),
            "--freeze-backbone-epochs",
            str(profile.freeze_backbone_epochs),
            "--loss-type",
            profile.loss_type,
            "--focal-gamma",
            str(profile.focal_gamma),
            "--class-sampling",
            profile.class_sampling,
            "--hard-class-boost",
            str(profile.hard_class_boost),
            "--device",
            profile.device,
        ]
        if profile.tta_flip:
            cmd.append("--tta-flip")
        run_cmd(cmd, repo_root)


def promote_best(repo_root: Path, python_exe: str) -> None:
    run_cmd([python_exe, "scripts/promote_best_vit.py"], repo_root)


def regenerate_plots(repo_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        print(f"[warn] matplotlib not available, skipping plots: {exc}", flush=True)
        return

    metrics_path = repo_root / "website/models/metrics.json"
    confusion_path = repo_root / "website/models/confusion_matrix.csv"
    if not metrics_path.exists() or not confusion_path.exists():
        print("[warn] promoted metrics/confusion not found, skipping plots", flush=True)
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plot_dir = repo_root / "docs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    classes = list(metrics["per_class"].keys())
    precision = [metrics["per_class"][name]["precision"] for name in classes]
    recall = [metrics["per_class"][name]["recall"] for name in classes]

    x_axis = np.arange(len(classes))
    width = 0.38

    plt.figure(figsize=(11, 5))
    plt.bar(x_axis - width / 2, precision, width, label="Precision", color="#0f766e")
    plt.bar(x_axis + width / 2, recall, width, label="Recall", color="#0ea5e9")
    plt.xticks(x_axis, classes, rotation=25)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Class-wise Precision and Recall (Promoted ViT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "precision_recall_by_class.png", dpi=160)
    plt.close()

    pred_counts = {row["class_key"]: row["predicted_count"] for row in metrics["prediction_counts"]}
    true_counts = {row["class_key"]: row["true_count"] for row in metrics["prediction_counts"]}

    true_vals = [true_counts[name] for name in classes]
    pred_vals = [pred_counts[name] for name in classes]

    plt.figure(figsize=(11, 5))
    plt.bar(x_axis - width / 2, true_vals, width, label="True Count", color="#334155")
    plt.bar(x_axis + width / 2, pred_vals, width, label="Predicted Count", color="#16a34a")
    plt.xticks(x_axis, classes, rotation=25)
    plt.ylabel("Count")
    plt.title("True vs Predicted Counts by Class (Promoted ViT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "true_vs_predicted_counts.png", dpi=160)
    plt.close()

    rows = []
    with confusion_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        labels = header[1:]
        for row in reader:
            rows.append([int(value) for value in row[1:]])

    confusion = np.array(rows)
    plt.figure(figsize=(8, 7))
    plt.imshow(confusion, cmap="YlGnBu")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=30)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap (Promoted ViT)")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(plot_dir / "confusion_matrix_heatmap.png", dpi=160)
    plt.close()

    print("[report] plots updated", flush=True)


def summarize_runs(repo_root: Path) -> None:
    artifacts_dir = repo_root / "artifacts"
    rows = []
    for run_dir in sorted(artifacts_dir.glob("vit_*")):
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_val = None
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            val_values = history.get("val_accuracy", [])
            if val_values:
                best_val = float(max(val_values))

        rows.append(
            {
                "run": run_dir.name,
                "test_acc": float(metrics.get("accuracy", -1.0)),
                "macro_p": float(metrics.get("macro_precision", -1.0)),
                "macro_r": float(metrics.get("macro_recall", -1.0)),
                "best_val": best_val,
            }
        )

    rows.sort(key=lambda item: item["test_acc"], reverse=True)

    print("[report] vit run summary", flush=True)
    print("run,test_acc,macro_p,macro_r,best_val", flush=True)
    for row in rows:
        best_val_text = "n/a" if row["best_val"] is None else f"{row['best_val']:.6f}"
        print(
            f"{row['run']},{row['test_acc']:.6f},{row['macro_p']:.6f},{row['macro_r']:.6f},{best_val_text}",
            flush=True,
        )


def validate_runtime(repo_root: Path) -> None:
    main_path = repo_root / "website/main.py"
    spec = importlib.util.spec_from_file_location("main_mod", str(main_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import runtime module from {main_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.RUNTIME
    print("[validate] backend=", runtime.get("backend"), flush=True)
    print("[validate] model_path=", runtime.get("model_path"), flush=True)
    print("[validate] image_size=", runtime.get("image_size"), flush=True)
    print("[validate] classes=", len(runtime.get("classes", [])), flush=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    if args.mode in ("full", "train"):
        profiles = selected_profiles(args.profiles)
        train_profiles(repo_root, args.python, profiles, args.skip_existing)

    if args.mode in ("full", "promote"):
        promote_best(repo_root, args.python)

    if args.mode in ("full", "report"):
        summarize_runs(repo_root)
        regenerate_plots(repo_root)

    if args.mode in ("full", "validate"):
        validate_runtime(repo_root)

    print("[done] pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
