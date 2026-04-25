from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def build_annotation_index(coco_data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images_by_id = {int(img["id"]): img for img in coco_data["images"]}
    annots_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        annots_by_image[int(ann["image_id"])].append(ann)
    return images_by_id, annots_by_image


def build_category_metadata(coco_data: dict[str, Any]) -> tuple[list[int], dict[int, int]]:
    categories = {int(cat["id"]): str(cat["name"]) for cat in coco_data["categories"]}
    tooth_ids = sorted(cid for cid, name in categories.items() if name.isdigit() and 1 <= int(name) <= 32)
    if not tooth_ids:
        raise ValueError("No tooth categories 1..32 were found in COCO categories.")
    tooth_label_map = {category_id: idx + 1 for idx, category_id in enumerate(tooth_ids)}
    return tooth_ids, tooth_label_map


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


def is_valid_xyxy(box: list[float], min_size: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def build_detection_records(coco_data: dict[str, Any], image_dirs: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    images_by_id, annots_by_image = build_annotation_index(coco_data)
    tooth_ids, tooth_label_map = build_category_metadata(coco_data)

    records: list[dict[str, Any]] = []
    for image_id, image_info in images_by_id.items():
        width = int(image_info["width"])
        height = int(image_info["height"])
        subset = image_info.get("subset", "train")
        image_path = image_dirs[subset] / image_info["file_name"]

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in annots_by_image.get(image_id, []):
            category_id = int(ann["category_id"])
            if category_id not in tooth_ids:
                continue
            box = clip_box_xyxy(coco_bbox_to_xyxy(ann["bbox"]), width, height)
            if is_valid_xyxy(box):
                boxes.append(box)
                labels.append(tooth_label_map[category_id])

        if boxes:
            records.append(
                {
                    "image_id": image_id,
                    "group_id": f"{subset}:{image_info['file_name']}",
                    "subset": subset,
                    "image_path": str(image_path),
                    "file_name": image_info["file_name"],
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
            )
    return records, tooth_label_map


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
