from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def build_annotation_index(
    coco_data: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images_by_id = {int(img["id"]): img for img in coco_data["images"]}
    annots_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        annots_by_image[int(ann["image_id"])].append(ann)
    return images_by_id, annots_by_image


def build_category_metadata(
    coco_data: dict[str, Any],
) -> tuple[dict[int, str], list[int], int, dict[int, int]]:
    categories = {int(cat["id"]): str(cat["name"]) for cat in coco_data["categories"]}

    tooth_ids = sorted(
        cid for cid, name in categories.items() if name.isdigit() and 1 <= int(name) <= 32
    )
    if not tooth_ids:
        raise ValueError("No tooth categories 1..32 were found in COCO categories.")

    caries_id = next((cid for cid, name in categories.items() if "caries" in name.lower()), None)
    if caries_id is None:
        raise ValueError("No 'caries' category found in COCO categories.")

    tooth_label_map = {category_id: idx + 1 for idx, category_id in enumerate(tooth_ids)}
    return categories, tooth_ids, caries_id, tooth_label_map


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


def build_module1_detection_records(
    coco_data: dict[str, Any],
    raw_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    raw_dir = Path(raw_dir)
    images_by_id, annots_by_image = build_annotation_index(coco_data)
    _, tooth_ids, _, tooth_label_map = build_category_metadata(coco_data)

    records: list[dict[str, Any]] = []
    for image_id, img_info in images_by_id.items():
        width = int(img_info["width"])
        height = int(img_info["height"])
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
                    "group_id": image_id,
                    "image_path": str(raw_dir / img_info["file_name"]),
                    "file_name": img_info["file_name"],
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
    stratify_key: Optional[str] = None,
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

    all_groups = list(groups_to_records)
    if len(all_groups) < 3:
        raise ValueError("Need at least 3 groups to build train/val/test splits.")

    def stratify(groups: list[Any]) -> Optional[list[int]]:
        if stratify_key is None:
            return None
        labels = [max(int(rec[stratify_key]) for rec in groups_to_records[g]) for g in groups]
        return labels if len(set(labels)) > 1 else None

    train_val_groups, test_groups = train_test_split(
        all_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify(all_groups),
    )
    relative_val_ratio = val_size / (train_size + val_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=stratify(train_val_groups),
    )

    def flatten(groups: list[Any]) -> list[dict[str, Any]]:
        return [record for group in groups for record in groups_to_records[group]]

    return flatten(train_groups), flatten(val_groups), flatten(test_groups)


def summarize_binary_labels(records: list[dict[str, Any]], label_key: str = "label") -> dict[str, int]:
    labels = [int(record[label_key]) for record in records]
    positives = int(sum(labels))
    return {"total": len(labels), "positive": positives, "negative": len(labels) - positives}
