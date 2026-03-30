from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboflow import Roboflow


@dataclass(frozen=True)
class ClassificationDownloadConfig:
    workspace: str = "wishis64"
    project_name: str = "se-iwfnq"
    version: int = 1
    export_format: str = "coco-segmentation"
    data_root: Path = Path("data") / "classification"
    api_key: str | None = None

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not key:
            raise ValueError("Roboflow API key not found. Set ROBOFLOW_API_KEY or pass api_key.")
        return key


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_coco_splits(dataset_root: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    json_files = sorted(dataset_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No COCO/COCO-segmentation JSON files found under {dataset_root}")

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    merged_categories: list[dict[str, Any]] | None = None
    image_id_offset = 0
    annotation_id_offset = 0
    image_dirs: dict[str, Path] = {}

    for json_path in json_files:
        subset_name = json_path.parent.name.lower()
        coco = _load_json(json_path)
        if merged_categories is None:
            merged_categories = coco.get("categories", [])

        original_to_new_image_id: dict[int, int] = {}
        for image in coco.get("images", []):
            new_image = dict(image)
            old_id = int(image["id"])
            new_id = old_id + image_id_offset
            new_image["id"] = new_id
            new_image["subset"] = subset_name
            merged_images.append(new_image)
            original_to_new_image_id[old_id] = new_id

        for ann in coco.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = int(ann["id"]) + annotation_id_offset
            new_ann["image_id"] = original_to_new_image_id[int(ann["image_id"])]
            merged_annotations.append(new_ann)

        if coco.get("images"):
            image_id_offset = max(int(img["id"]) for img in merged_images) + 1
        if coco.get("annotations"):
            annotation_id_offset = max(int(ann["id"]) for ann in merged_annotations) + 1
        image_dirs[subset_name] = json_path.parent

    return dataset_root, {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories or [],
    }, image_dirs


def _has_coco_json(dataset_root: Path) -> bool:
    return dataset_root.exists() and any(dataset_root.rglob("*.json"))


def download_classification_dataset(config: ClassificationDownloadConfig) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    rf = Roboflow(api_key=config.resolved_api_key())
    project = rf.workspace(config.workspace).project(config.project_name)
    version = project.version(config.version)
    download_dir = config.data_root.resolve()
    dataset = version.download(model_format=config.export_format, location=str(download_dir), overwrite=True)
    dataset_root = Path(dataset.location)
    return _merge_coco_splits(dataset_root)


def load_or_download_classification_dataset(
    config: ClassificationDownloadConfig, *, force_download: bool = False
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    dataset_root = config.data_root.resolve()
    if not force_download and _has_coco_json(dataset_root):
        return _merge_coco_splits(dataset_root)
    return download_classification_dataset(config)
