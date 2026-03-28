from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import train_test_split


def build_annotation_index(coco_data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    images_by_id = {int(img["id"]): img for img in coco_data["images"]}
    annots_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        annots_by_image[int(ann["image_id"])].append(ann)
    categories = {int(cat["id"]): str(cat["name"]) for cat in coco_data["categories"]}
    return images_by_id, annots_by_image, categories


def _bbox_from_segmentation(segmentation: Any) -> list[float] | None:
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        xs: list[float] = []
        ys: list[float] = []
        for polygon in segmentation:
            if not polygon:
                continue
            coords = list(map(float, polygon))
            xs.extend(coords[0::2])
            ys.extend(coords[1::2])
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def coco_bbox_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def clip_box_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    ]


def expand_box_xyxy(box: list[float], width: int, height: int, margin: float = 0.08) -> list[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    return [
        max(0, int(round(x1 - bw * margin))),
        max(0, int(round(y1 - bh * margin))),
        min(width, int(round(x2 + bw * margin))),
        min(height, int(round(y2 + bh * margin))),
    ]


def is_valid_xyxy(box: list[float], min_size: float = 8.0) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def _resolve_bbox(annotation: dict[str, Any], width: int, height: int) -> list[float] | None:
    box = _bbox_from_segmentation(annotation.get("segmentation"))
    if box is None and annotation.get("bbox"):
        box = coco_bbox_to_xyxy(annotation["bbox"])
    if box is None:
        return None
    box = clip_box_xyxy(box, width, height)
    return box if is_valid_xyxy(box) else None


def build_classification_records_from_masks(
    coco_data: dict[str, Any],
    image_dirs: dict[str, Path],
    *,
    positive_classes: Iterable[str] = ("Caries",),
    crop_margin: float = 0.08,
) -> list[dict[str, Any]]:
    images_by_id, annots_by_image, categories = build_annotation_index(coco_data)
    positive_names = {name.lower() for name in positive_classes}

    records: list[dict[str, Any]] = []
    for image_id, image_info in images_by_id.items():
        subset = image_info.get("subset", "train")
        width = int(image_info["width"])
        height = int(image_info["height"])
        image_path = image_dirs[subset] / image_info["file_name"]

        for ann in annots_by_image.get(image_id, []):
            class_name = categories[int(ann["category_id"])]
            if class_name.lower() not in positive_names:
                continue
            tooth_box = _resolve_bbox(ann, width, height)
            if tooth_box is None:
                continue
            records.append(
                {
                    "record_id": f"{image_id}_{ann['id']}",
                    "group_id": f"{subset}:{image_info['file_name']}",
                    "subset": subset,
                    "source_image_id": image_id,
                    "image_path": str(image_path),
                    "file_name": image_info["file_name"],
                    "crop_box": expand_box_xyxy(tooth_box, width, height, margin=crop_margin),
                    "tooth_box": tooth_box,
                    "label": 1,
                    "source_class": class_name,
                }
            )
    return records


def split_grouped_records(
    records: list[dict[str, Any]],
    *,
    group_key: str = "group_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    groups_to_records: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups_to_records[record[group_key]].append(record)

    groups = list(groups_to_records)
    train_val_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=random_state)
    relative_val_ratio = val_size / (train_size + val_size)
    train_groups, val_groups = train_test_split(train_val_groups, test_size=relative_val_ratio, random_state=random_state)

    def flatten(selected: list[Any]) -> list[dict[str, Any]]:
        return [record for group in selected for record in groups_to_records[group]]

    return flatten(train_groups), flatten(val_groups), flatten(test_groups)


def summarize_binary_labels(records: list[dict[str, Any]], label_key: str = "label") -> dict[str, int]:
    labels = [int(record[label_key]) for record in records]
    positives = int(sum(labels))
    return {"total": len(labels), "positive": positives, "negative": len(labels) - positives}
