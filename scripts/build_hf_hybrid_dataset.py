#!/usr/bin/env python3
"""Build a balanced 8-class dataset from Hugging Face dermatology datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

from datasets import load_dataset, load_dataset_builder
from PIL import Image


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

DERMNET_DATASET = "Muzmmillcoste/dermnet"
SKIN_CANCER_DATASET = "Pranavkpba2000/skin_cancer_small_dataset"

DERMNET_MAPPING = {
    "Acne and Rosacea Photos": "acne",
    "Atopic Dermatitis Photos": "eczema",
    "Eczema Photos": "eczema",
    "Poison Ivy Photos and other Contact Dermatitis": "eczema",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "tinea",
}

SKIN_CANCER_MAPPING = {
    "AK": "ak",
    "BCC": "bcc",
    "BKL": "bkl",
    "MEL": "melanoma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/hf_hybrid")
    parser.add_argument("--target-per-class", type=int, default=80)
    parser.add_argument("--min-side", type=int, default=96)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--profile",
        choices=("default", "strict"),
        default="default",
        help="default uses broader Dermnet category mapping; strict drops noisier eczema-adjacent sources.",
    )
    return parser.parse_args()


def pil_to_jpeg_bytes(image: Image.Image, min_side: int) -> tuple[bytes, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < min_side:
        raise ValueError(f"image too small: {width}x{height}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue(), width, height


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    total = len(paths)
    if total < 3:
        return paths, [], []
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)
    return paths[:train_end], paths[train_end:val_end], paths[val_end:]


def ensure_counts_complete(counts: dict[str, int], target_per_class: int) -> None:
    missing = {key: count for key, count in counts.items() if count < target_per_class}
    if missing:
        raise RuntimeError(f"dataset build incomplete: {missing}")


def save_manifest(output_dir: Path, manifest_rows: list[dict[str, str]], split_counts: dict[str, dict[str, int]], args: argparse.Namespace) -> None:
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_key",
                "app_label",
                "source_dataset",
                "source_split",
                "source_label",
                "source_index",
                "status",
                "saved_path",
                "sha256",
                "width",
                "height",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "created_at_epoch": time.time(),
        "source_datasets": [DERMNET_DATASET, SKIN_CANCER_DATASET],
        "class_order": CLASS_ORDER,
        "class_labels": APP_LABELS,
        "target_per_class": args.target_per_class,
        "profile": args.profile,
        "split_counts": split_counts,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def get_dermnet_mapping(profile: str) -> dict[str, str]:
    if profile == "strict":
        return {
            "Acne and Rosacea Photos": "acne",
            "Atopic Dermatitis Photos": "eczema",
            "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
            "Tinea Ringworm Candidiasis and other Fungal Infections": "tinea",
        }
    return DERMNET_MAPPING


def collect_dermnet(output_dir: Path, target_per_class: int, min_side: int, seed: int, profile: str) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    raw_dir = output_dir / "raw"
    counts = {key: 0 for key in CLASS_ORDER}
    sha_seen: set[str] = set()

    builder = load_dataset_builder(DERMNET_DATASET)
    label_names = builder.info.features["label"].names
    mapping = get_dermnet_mapping(profile)
    needed = set(mapping.values())

    print("[hf-hybrid] collecting Dermnet classes", flush=True)
    for split in ("train", "test"):
        print(f"[hf-hybrid] Dermnet split={split}", flush=True)
        dataset = load_dataset(DERMNET_DATASET, split=split, streaming=True)
        for index, row in enumerate(dataset):
            source_label = label_names[row["label"]]
            class_key = mapping.get(source_label)
            if class_key is None or class_key not in needed or counts[class_key] >= target_per_class:
                continue
            status = "failed"
            error = ""
            sha256 = ""
            width = height = 0
            saved_path = ""
            try:
                jpg_bytes, width, height = pil_to_jpeg_bytes(row["image"], min_side)
                sha256 = hashlib.sha256(jpg_bytes).hexdigest()
                if sha256 in sha_seen:
                    raise ValueError("duplicate image content")
                sha_seen.add(sha256)
                destination = raw_dir / class_key / f"{class_key}_{counts[class_key] + 1:04d}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(jpg_bytes)
                counts[class_key] += 1
                saved_path = str(destination.relative_to(output_dir))
                status = "saved"
                if counts[class_key] % 10 == 0 or counts[class_key] == target_per_class:
                    print(f"[hf-hybrid] Dermnet {class_key}: {counts[class_key]}/{target_per_class}", flush=True)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            manifest_rows.append(
                {
                    "class_key": class_key,
                    "app_label": APP_LABELS[class_key],
                    "source_dataset": DERMNET_DATASET,
                    "source_split": split,
                    "source_label": source_label,
                    "source_index": str(index),
                    "status": status,
                    "saved_path": saved_path,
                    "sha256": sha256,
                    "width": str(width),
                    "height": str(height),
                    "error": error,
                }
            )
            if all(counts[key] >= target_per_class for key in needed):
                print("[hf-hybrid] Dermnet collection complete", flush=True)
                return manifest_rows
    return manifest_rows


def collect_skin_cancer(output_dir: Path, target_per_class: int, min_side: int) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    raw_dir = output_dir / "raw"
    counts = {key: len(list((raw_dir / key).glob("*.jpg"))) for key in CLASS_ORDER}
    sha_seen = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for key in CLASS_ORDER
        for path in (raw_dir / key).glob("*.jpg")
    }
    needed = {"ak", "bcc", "bkl", "melanoma"}

    print("[hf-hybrid] collecting skin_cancer classes", flush=True)
    builder = load_dataset_builder(SKIN_CANCER_DATASET)
    label_names = builder.info.features["label"].names
    for split in ("train", "validation", "test"):
        if split not in builder.info.splits:
            continue
        print(f"[hf-hybrid] skin_cancer split={split}", flush=True)
        dataset = load_dataset(SKIN_CANCER_DATASET, split=split, streaming=True).shuffle(seed=19, buffer_size=256)
        for index, row in enumerate(dataset):
            source_label = label_names[row["label"]]
            class_key = SKIN_CANCER_MAPPING.get(source_label)
            if class_key is None or counts[class_key] >= target_per_class:
                continue
            status = "failed"
            error = ""
            sha256 = ""
            width = height = 0
            saved_path = ""
            try:
                jpg_bytes, width, height = pil_to_jpeg_bytes(row["image"], min_side)
                sha256 = hashlib.sha256(jpg_bytes).hexdigest()
                if sha256 in sha_seen:
                    raise ValueError("duplicate image content")
                sha_seen.add(sha256)
                destination = raw_dir / class_key / f"{class_key}_{counts[class_key] + 1:04d}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(jpg_bytes)
                counts[class_key] += 1
                saved_path = str(destination.relative_to(output_dir))
                status = "saved"
                if counts[class_key] % 10 == 0 or counts[class_key] == target_per_class:
                    print(f"[hf-hybrid] skin_cancer {class_key}: {counts[class_key]}/{target_per_class}", flush=True)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            manifest_rows.append(
                {
                    "class_key": class_key,
                    "app_label": APP_LABELS[class_key],
                    "source_dataset": SKIN_CANCER_DATASET,
                    "source_split": split,
                    "source_label": source_label,
                    "source_index": str(index),
                    "status": status,
                    "saved_path": saved_path,
                    "sha256": sha256,
                    "width": str(width),
                    "height": str(height),
                    "error": error,
                }
            )
            if all(counts[key] >= target_per_class for key in needed):
                print("[hf-hybrid] skin_cancer collection complete", flush=True)
                return manifest_rows
    return manifest_rows


def build_splits(output_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> dict[str, dict[str, int]]:
    splits_dir = output_dir / "splits"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        (splits_dir / split_name).mkdir(parents=True, exist_ok=True)

    for class_key in CLASS_ORDER:
        files = sorted((output_dir / "raw" / class_key).glob("*.jpg"))
        train_files, val_files, test_files = split_paths(files, train_ratio, val_ratio, seed)
        split_counts[class_key] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }
        for split_name, items in (
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        ):
            destination_dir = splits_dir / split_name / class_key
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in items:
                shutil.copy2(source, destination_dir / source.name)
    return split_counts


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    manifest_rows.extend(collect_dermnet(output_dir, args.target_per_class, args.min_side, args.seed, args.profile))
    manifest_rows.extend(collect_skin_cancer(output_dir, args.target_per_class, args.min_side))

    raw_counts = {
        class_key: len(list((output_dir / "raw" / class_key).glob("*.jpg")))
        for class_key in CLASS_ORDER
    }
    ensure_counts_complete(raw_counts, args.target_per_class)
    split_counts = build_splits(output_dir, args.train_ratio, args.val_ratio, args.seed)
    save_manifest(output_dir, manifest_rows, split_counts, args)

    print("[hf-hybrid] build complete")
    for class_key in CLASS_ORDER:
        counts = split_counts[class_key]
        print(
            f"  - {class_key}: raw={raw_counts[class_key]} "
            f"train={counts['train']} val={counts['validation']} test={counts['test']}"
        )
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
