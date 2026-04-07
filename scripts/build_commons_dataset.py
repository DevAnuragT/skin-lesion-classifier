#!/usr/bin/env python3
"""Build a dermatology dataset from Wikimedia Commons file search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 Codex Wikimedia Dataset Builder"
BLOCKED_TITLE_TERMS = (
    "histopathology",
    "micrograph",
    "staining",
    "diagram",
    "pattern",
    "schema",
    "illustration",
    "icon",
    "logo",
    "clinical features",
    "dermoscopy",
    "drawing",
    "x-ray",
)
ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class ClassSpec:
    key: str
    app_label: str
    search_terms: tuple[str, ...]
    categories: tuple[str, ...]


CLASS_SPECS = (
    ClassSpec(
        "acne",
        "Acne",
        ("acne vulgaris", "cystic acne", "acne"),
        ("Category:Acne", "Category:Cystic acne"),
    ),
    ClassSpec(
        "bcc",
        "Basal cell carcinoma",
        (
            "basal cell carcinoma",
            "basal-cell carcinoma",
            "nodular basal cell carcinoma",
            "superficial basal cell carcinoma",
            "morpheaform basal cell carcinoma",
            "pigmented basal cell carcinoma",
        ),
        ("Category:Basal-cell carcinoma",),
    ),
    ClassSpec(
        "bkl",
        "Benign Keratosis-like Lesions (BKL)",
        ("seborrheic keratosis", "solar lentigo", "age spot", "liver spot"),
        ("Category:Seborrheic keratosis", "Category:Solar lentigo"),
    ),
    ClassSpec(
        "eczema",
        "Atopic dermatitis(Eczema)",
        ("atopic dermatitis", "eczema", "contact dermatitis", "nummular dermatitis"),
        ("Category:Atopic dermatitis",),
    ),
    ClassSpec(
        "ak",
        "Actinic keratosis(AK)",
        ("actinic keratosis", "actinic keratoses"),
        ("Category:Actinic keratosis",),
    ),
    ClassSpec(
        "melanoma",
        "Melanoma",
        ("cutaneous melanoma", "melanoma"),
        ("Category:Melanoma",),
    ),
    ClassSpec(
        "psoriasis",
        "Psoriasis",
        ("plaque psoriasis", "psoriasis", "guttate psoriasis"),
        ("Category:Psoriasis", "Category:Guttate psoriasis"),
    ),
    ClassSpec(
        "tinea",
        "Tinea(Ringworm)",
        ("tinea corporis", "tinea faciei", "tinea capitis", "tinea manuum", "ringworm", "dermatophytosis"),
        ("Category:Dermatophytosis", "Category:Tinea"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/commons_auto")
    parser.add_argument("--target-per-class", type=int, default=40)
    parser.add_argument("--search-limit-per-term", type=int, default=120)
    parser.add_argument("--category-limit", type=int, default=120)
    parser.add_argument("--min-side", type=int, default=96)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def api_get(params: dict[str, str], timeout: int) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"commons api request failed: {url}") from last_error


def normalize_title(title: str) -> str:
    lowered = title.lower()
    return lowered.removeprefix("file:")


def title_allowed(title: str) -> bool:
    normalized = normalize_title(title)
    if not normalized.endswith(ALLOWED_SUFFIXES):
        return False
    return not any(term in normalized for term in BLOCKED_TITLE_TERMS)


def search_file_titles(term: str, limit: int, timeout: int) -> list[str]:
    titles: list[str] = []
    offset = 0
    while len(titles) < limit:
        batch_size = min(50, limit - len(titles))
        data = api_get(
            {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srnamespace": "6",
                "srlimit": str(batch_size),
                "sroffset": str(offset),
                "format": "json",
            },
            timeout,
        )
        batch = [item["title"] for item in data["query"]["search"]]
        if not batch:
            break
        titles.extend(batch)
        offset += len(batch)
    return titles


def category_file_titles(category: str, limit: int, timeout: int) -> list[str]:
    titles: list[str] = []
    continuation: str | None = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": str(min(50, limit - len(titles))),
            "format": "json",
        }
        if continuation:
            params["cmcontinue"] = continuation
        data = api_get(params, timeout)
        batch = [item["title"] for item in data["query"]["categorymembers"]]
        if not batch:
            break
        titles.extend(batch)
        continuation = data.get("continue", {}).get("cmcontinue")
        if not continuation:
            break
    return titles


def image_info(titles: list[str], timeout: int) -> dict[str, dict[str, str]]:
    chunks = [titles[index : index + 10] for index in range(0, len(titles), 10)]
    info: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        data = api_get(
            {
                "action": "query",
                "prop": "imageinfo",
                "titles": "|".join(chunk),
                "iiprop": "url|mime",
                "format": "json",
            },
            timeout,
        )
        pages = data["query"]["pages"].values()
        for page in pages:
            title = page.get("title")
            imageinfo = page.get("imageinfo")
            if title and imageinfo:
                info[title] = {
                    "url": imageinfo[0].get("url", ""),
                    "mime": imageinfo[0].get("mime", ""),
                }
    return info


def collect_candidates(spec: ClassSpec, args: argparse.Namespace) -> list[dict[str, str]]:
    titles: list[tuple[str, str]] = []
    seen_titles: set[str] = set()

    for term in spec.search_terms:
        try:
            results = search_file_titles(term, args.search_limit_per_term, args.timeout)
        except RuntimeError:
            continue
        for title in results:
            if title in seen_titles or not title_allowed(title):
                continue
            seen_titles.add(title)
            titles.append((title, f"search:{term}"))

    for category in spec.categories:
        try:
            results = category_file_titles(category, args.category_limit, args.timeout)
        except RuntimeError:
            continue
        for title in results:
            if title in seen_titles or not title_allowed(title):
                continue
            seen_titles.add(title)
            titles.append((title, f"category:{category}"))

    try:
        metadata = image_info([title for title, _ in titles], args.timeout)
    except RuntimeError:
        metadata = {}
    candidates = []
    for title, source in titles:
        item = metadata.get(title)
        if not item or not item["mime"].startswith("image/"):
            continue
        candidates.append(
            {
                "title": title,
                "source": source,
                "url": item["url"],
                "mime": item["mime"],
            }
        )
    return candidates


def download_and_convert(url: str, timeout: int, min_side: int) -> tuple[bytes, int, int]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    else:
        raise RuntimeError(f"image download failed: {url}") from last_error
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < min_side:
            raise ValueError(f"image too small: {width}x{height}")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue(), width, height


def split_files(files: list[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    total = len(files)
    if total < 3:
        return files, [], []
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total - 1)
    return files[:train_end], files[train_end:val_end], files[val_end:]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    raw_counts: dict[str, int] = {}
    sha_seen: set[str] = set()

    for spec in CLASS_SPECS:
        class_dir = raw_dir / spec.key
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"[commons] collecting {spec.key}")
        candidates = collect_candidates(spec, args)
        saved = 0
        for candidate in candidates:
            if saved >= args.target_per_class:
                break
            status = "failed"
            error = ""
            width = height = 0
            sha256 = ""
            saved_path = ""
            try:
                jpg_bytes, width, height = download_and_convert(candidate["url"], args.timeout, args.min_side)
                sha256 = hashlib.sha256(jpg_bytes).hexdigest()
                if sha256 in sha_seen:
                    raise ValueError("duplicate image content")
                sha_seen.add(sha256)
                destination = class_dir / f"{spec.key}_{saved + 1:04d}.jpg"
                destination.write_bytes(jpg_bytes)
                saved += 1
                saved_path = str(destination.relative_to(output_dir))
                status = "saved"
            except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as exc:
                error = str(exc)
            manifest_rows.append(
                {
                    "class_key": spec.key,
                    "app_label": spec.app_label,
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "url": candidate["url"],
                    "mime": candidate["mime"],
                    "status": status,
                    "saved_path": saved_path,
                    "sha256": sha256,
                    "width": str(width),
                    "height": str(height),
                    "error": error,
                }
            )
        raw_counts[spec.key] = saved
        print(f"[commons] {spec.key}: saved {saved} images from {len(candidates)} live candidates")

    splits_dir = output_dir / "splits"
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        (splits_dir / split_name).mkdir(parents=True, exist_ok=True)
    for spec in CLASS_SPECS:
        files = sorted((raw_dir / spec.key).glob("*.jpg"))
        train_files, val_files, test_files = split_files(files, args.train_ratio, args.val_ratio, args.seed)
        split_counts[spec.key] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }
        for split_name, items in (
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        ):
            destination_dir = splits_dir / split_name / spec.key
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in items:
                shutil.copy2(source, destination_dir / source.name)

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_key",
                "app_label",
                "title",
                "source",
                "url",
                "mime",
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
        "source": "wikimedia-commons",
        "class_order": [spec.key for spec in CLASS_SPECS],
        "class_labels": {spec.key: spec.app_label for spec in CLASS_SPECS},
        "raw_counts": raw_counts,
        "split_counts": split_counts,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[commons] build complete")
    for spec in CLASS_SPECS:
        counts = split_counts[spec.key]
        print(
            f"  - {spec.key}: raw={raw_counts[spec.key]} "
            f"train={counts['train']} val={counts['validation']} test={counts['test']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
