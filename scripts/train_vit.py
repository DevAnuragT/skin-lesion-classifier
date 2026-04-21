#!/usr/bin/env python3
"""Train a Vision Transformer skin lesion classifier."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SkinFolderDataset(Dataset):
    def __init__(self, split_dir: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for index, class_key in enumerate(CLASS_ORDER):
            class_dir = split_dir / class_key
            if not class_dir.exists():
                raise FileNotFoundError(f"missing class directory: {class_dir}")
            image_paths = [
                path for path in sorted(class_dir.iterdir()) if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]
            if not image_paths:
                raise RuntimeError(f"no images found in {class_dir}")
            self.samples.extend((image_path, index) for image_path in image_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/hf_hybrid_strict_80/splits")
    parser.add_argument("--output-dir", default="artifacts/vit_hf_strict_80")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--aug-level", choices=("light", "medium", "strong"), default="medium")
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-start-epoch", type=int, default=3)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--loss-type", choices=("ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-sampling", choices=("uniform", "hard"), default="uniform")
    parser.add_argument("--hard-class-boost", type=float, default=1.0)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int, aug_level: str, random_erasing: float) -> tuple[transforms.Compose, transforms.Compose]:
    if aug_level == "light":
        train_ops = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.01),
        ]
    elif aug_level == "strong":
        train_ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.12),
            transforms.RandAugment(num_ops=2, magnitude=8),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        ]
    else:
        train_ops = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ]

    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if random_erasing > 0:
        train_ops.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"))

    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = make_transforms(args.image_size, args.aug_level, args.random_erasing)
    data_root = Path(args.data_root)
    train_ds = SkinFolderDataset(data_root / "train", train_transform)
    val_ds = SkinFolderDataset(data_root / "validation", eval_transform)
    test_ds = SkinFolderDataset(data_root / "test", eval_transform)

    train_sampler = None
    train_shuffle = True
    if args.class_sampling == "hard":
        hard_classes = {"tinea", "eczema", "psoriasis", "acne"}
        sample_weights = []
        for _, label in train_ds.samples:
            class_key = CLASS_ORDER[label]
            weight = args.hard_class_boost if class_key in hard_classes else 1.0
            sample_weights.append(weight)
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("head"):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixup_fn: Mixup | None,
    ema_model: ModelEmaV2 | None,
) -> tuple[float, float]:
    model.train()
    losses = []
    accuracies = []
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        hard_labels = labels

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        losses.append(float(loss.item()))
        accuracies.append(accuracy_from_logits(logits.detach(), hard_labels))

    return float(np.mean(losses)), float(np.mean(accuracies))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
    tta_flip: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    losses = []
    accuracies = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if tta_flip:
                flipped_logits = model(torch.flip(images, dims=[3]))
                logits = (logits + flipped_logits) * 0.5
            loss = criterion(logits, labels)

            losses.append(float(loss.item()))
            accuracies.append(accuracy_from_logits(logits, labels))

            if collect_predictions:
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

    return float(np.mean(losses)), float(np.mean(accuracies)), y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int]) -> tuple[dict[str, object], np.ndarray]:
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[true_idx, pred_idx] += 1

    per_class: dict[str, dict[str, float | int | str]] = {}
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

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_precision = float(np.mean([per_class[class_key]["precision"] for class_key in CLASS_ORDER]))
    macro_recall = float(np.mean([per_class[class_key]["recall"] for class_key in CLASS_ORDER]))

    prediction_counts = []
    for index, class_key in enumerate(CLASS_ORDER):
        prediction_counts.append(
            {
                "class_key": class_key,
                "app_label": APP_LABELS[class_key],
                "predicted_count": int((np.array(y_pred) == index).sum()),
                "true_count": int((np.array(y_true) == index).sum()),
            }
        )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "prediction_counts": prediction_counts,
    }
    return metrics, confusion


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    model: nn.Module,
    history: dict[str, list[float]],
    metrics: dict[str, object],
    confusion: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": args.model_name,
            "image_size": args.image_size,
            "class_order": CLASS_ORDER,
            "state_dict": model.state_dict(),
        },
        output_dir / "best_model.pt",
    )

    (output_dir / "class_names.json").write_text(
        json.dumps(
            {
                "class_order": CLASS_ORDER,
                "app_labels": APP_LABELS,
                "image_size": args.image_size,
                "model_name": args.model_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_ORDER])
        for class_key, row in zip(CLASS_ORDER, confusion):
            writer.writerow([class_key, *map(int, row.tolist())])


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=len(CLASS_ORDER),
        img_size=args.image_size,
        drop_rate=0.2,
        drop_path_rate=args.drop_path_rate,
    )
    model.to(device)

    mixup_fn: Mixup | None = None
    if args.mixup_prob > 0 and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=len(CLASS_ORDER),
        )

    if mixup_fn is not None:
        criterion: nn.Module = SoftTargetCrossEntropy()
    elif args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema_model = ModelEmaV2(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_model_source = "model"
    best_val_accuracy = -1.0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                set_backbone_trainable(model, trainable=False)
            elif epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, trainable=True)
                print(f"[training] backbone unfrozen at epoch {epoch}", flush=True)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, mixup_fn, ema_model)
        use_ema_for_eval = ema_model is not None and epoch >= max(1, args.ema_start_epoch)
        eval_model = ema_model.module if use_ema_for_eval else model
        val_loss, val_acc, _, _ = evaluate(eval_model, val_loader, eval_criterion, device, tta_flip=args.tta_flip)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_accuracy + 1e-4:
            best_val_accuracy = val_acc
            best_state = copy.deepcopy(eval_model.state_dict())
            best_model_source = "ema" if use_ema_for_eval else "model"
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[training] early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    _, _, y_true, y_pred = evaluate(model, test_loader, eval_criterion, device, collect_predictions=True, tta_flip=args.tta_flip)
    metrics, confusion = compute_metrics(y_true, y_pred)
    metrics["best_model_source"] = best_model_source

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, args, model, history, metrics, confusion)

    print("[training] complete", flush=True)
    print(json.dumps({"best_val_accuracy": best_val_accuracy, "test_accuracy": metrics["accuracy"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
