#!/usr/bin/env python3
"""Build a reproducible local dataset from the repo CSV of image URLs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"


@dataclass(frozen=True)
class ClassSpec:
    key: str
    app_label: str
    matcher_terms: tuple[str, ...]


CLASS_SPECS = (
    ClassSpec("acne", "Acne", ("acne",)),
    ClassSpec(
        "bcc",
        "Basal cell carcinoma",
        (
            "bcc",
            "basal cell carcinoma",
            "basal cell epithelioma",
            "basalioma",
            "rodent ulcer",
        ),
    ),
    ClassSpec(
        "bkl",
        "Benign Keratosis-like Lesions (BKL)",
        (
            "seborrhoeic keratosis",
            "seborrheic keratosis",
            "solar lentigo",
            "seborrhoeic wart",
            "seborrheic wart",
        ),
    ),
    ClassSpec(
        "eczema",
        "Atopic dermatitis(Eczema)",
        (
            "eczema",
            "dermatitis",
            "besnier prurigo",
            "pompholyx",
            "nummular",
            "discoid eczema",
        ),
    ),
    ClassSpec(
        "ak",
        "Actinic keratosis(AK)",
        (
            "actinic keratosis",
            "solar keratosis",
        ),
    ),
    ClassSpec("melanoma", "Melanoma", ("melanoma",)),
    ClassSpec(
        "psoriasis",
        "Psoriasis",
        (
            "psoriasis",
            "palmoplantar pustulosis",
        ),
    ),
    ClassSpec(
        "tinea",
        "Tinea(Ringworm)",
        (
            "tinea",
            "ringworm",
            "fungal skin infection",
        ),
    ),
)

# Priority matters for overlapping terms like "lentigo maligna melanoma".
CLASS_PRIORITY = ("melanoma", "bcc", "ak", "psoriasis", "eczema", "tinea", "acne", "bkl")
CLASS_BY_KEY = {spec.key: spec for spec in CLASS_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Data/data1-294.csv")
    parser.add_argument("--output", default="datasets/derma_auto")
    parser.add_argument("--target-per-class", type=int, default=140)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def classify_disorder(name: str) -> str | None:
    lowered = name.strip().lower()
    for key in CLASS_PRIORITY:
        spec = CLASS_BY_KEY[key]
        if any(term in lowered for term in spec.matcher_terms):
            return key
    return None


def load_candidates(csv_path: Path, seed: int) -> dict[str, list[dict[str, str]]]:
    grouped = {spec.key: [] for spec in CLASS_SPECS}
    seen_urls: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disorder = row["skin_disorder_name"].strip()
            url = row["images"].strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            class_key = classify_disorder(disorder)
            if class_key is None:
                continue
            grouped[class_key].append(
                {
                    "class_key": class_key,
                    "disorder": disorder,
                    "url": url,
                }
            )

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)
    return grouped


def download_bytes(url: str, timeout: int) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
    return body, content_type


def validate_and_convert_image(raw_bytes: bytes, min_side: int) -> tuple[bytes, int, int, str]:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < min_side:
            raise ValueError(f"image too small: {width}x{height}")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue(), width, height, "image/jpeg"


def build_raw_dataset(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(Path(args.csv), args.seed)
    counts = {spec.key: 0 for spec in CLASS_SPECS}
    sha_seen: set[str] = set()
    manifest_rows: list[dict[str, str]] = []

    for spec in CLASS_SPECS:
        class_dir = raw_dir / spec.key
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] collecting {spec.key} images")
        for candidate in candidates[spec.key]:
            if counts[spec.key] >= args.target_per_class:
                break

            error_message = ""
            width = height = 0
            content_type = ""
            saved_path = ""
            sha256 = ""
            status = "failed"
            try:
                raw_bytes, content_type = download_bytes(candidate["url"], args.timeout)
                if not content_type.lower().startswith("image/"):
                    raise ValueError(f"non-image content-type: {content_type or 'missing'}")
                jpg_bytes, width, height, content_type = validate_and_convert_image(raw_bytes, args.min_side)
                sha256 = hashlib.sha256(jpg_bytes).hexdigest()
                if sha256 in sha_seen:
                    raise ValueError("duplicate image content")
                sha_seen.add(sha256)
                file_name = f"{spec.key}_{counts[spec.key] + 1:04d}.jpg"
                destination = class_dir / file_name
                destination.write_bytes(jpg_bytes)
                counts[spec.key] += 1
                saved_path = str(destination.relative_to(output_dir))
                status = "saved"
            except (urllib.error.URLError, TimeoutError, ValueError, UnidentifiedImageError, OSError) as exc:
                error_message = str(exc)

            manifest_rows.append(
                {
                    "class_key": spec.key,
                    "app_label": spec.app_label,
                    "disorder": candidate["disorder"],
                    "url": candidate["url"],
                    "status": status,
                    "saved_path": saved_path,
                    "sha256": sha256,
                    "width": str(width),
                    "height": str(height),
                    "content_type": content_type,
                    "error": error_message,
                }
            )

        print(f"[dataset] {spec.key}: saved {counts[spec.key]} images")

    return manifest_rows, counts


def split_class_images(class_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    images = sorted(path for path in class_dir.iterdir() if path.is_file())
    rng = random.Random(seed)
    rng.shuffle(images)
    total = len(images)
    if total < 3:
        return images, [], []

    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def build_splits(args: argparse.Namespace, output_dir: Path) -> dict[str, dict[str, int]]:
    splits_dir = output_dir / "splits"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        (splits_dir / split_name).mkdir(parents=True, exist_ok=True)

    for spec in CLASS_SPECS:
        class_dir = output_dir / "raw" / spec.key
        train_files, val_files, test_files = split_class_images(
            class_dir, args.train_ratio, args.val_ratio, args.seed
        )
        split_counts[spec.key] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }
        for split_name, files in (
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        ):
            destination_dir = splits_dir / split_name / spec.key
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in files:
                shutil.copy2(source, destination_dir / source.name)
    return split_counts


def write_metadata(output_dir: Path, manifest_rows: list[dict[str, str]], raw_counts: dict[str, int], split_counts: dict[str, dict[str, int]], args: argparse.Namespace) -> None:
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_key",
                "app_label",
                "disorder",
                "url",
                "status",
                "saved_path",
                "sha256",
                "width",
                "height",
                "content_type",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "created_at_epoch": time.time(),
        "source_csv": args.csv,
        "target_per_class": args.target_per_class,
        "class_order": [spec.key for spec in CLASS_SPECS],
        "class_labels": {spec.key: spec.app_label for spec in CLASS_SPECS},
        "raw_counts": raw_counts,
        "split_counts": split_counts,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train + validation ratio must be < 1")

    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows, raw_counts = build_raw_dataset(args, output_dir)
    split_counts = build_splits(args, output_dir)
    write_metadata(output_dir, manifest_rows, raw_counts, split_counts, args)

    print("[dataset] build complete")
    for spec in CLASS_SPECS:
        splits = split_counts[spec.key]
        print(
            f"  - {spec.key}: raw={raw_counts[spec.key]} "
            f"train={splits['train']} val={splits['validation']} test={splits['test']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
