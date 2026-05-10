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


def _normalize_label_name(name: str) -> str:
    return str(name).strip().lower()


def _build_per_label_caps(
    label_names: Iterable[str],
    max_samples_per_label: int | dict[str, int] | None,
) -> dict[str, int | None]:
    names = list(label_names)
    if max_samples_per_label is None:
        return {name: None for name in names}
    if isinstance(max_samples_per_label, int):
        if max_samples_per_label <= 0:
            raise ValueError("max_samples_per_label must be > 0")
        return {name: max_samples_per_label for name in names}

    normalized_caps = {_normalize_label_name(name): int(value) for name, value in max_samples_per_label.items()}
    invalid = [name for name, value in normalized_caps.items() if value <= 0]
    if invalid:
        raise ValueError(f"max_samples_per_label values must be > 0, got invalid keys: {invalid}")

    caps_by_name: dict[str, int | None] = {}
    for name in names:
        caps_by_name[name] = normalized_caps.get(_normalize_label_name(name))
    return caps_by_name


def build_multiclass_classification_records_from_masks(
    coco_data: dict[str, Any],
    image_dirs: dict[str, Path],
    *,
    include_classes: Iterable[str] | None = None,
    crop_margin: float = 0.08,
    max_samples_per_label: int | dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    images_by_id, annots_by_image, categories = build_annotation_index(coco_data)

    include_names = None
    if include_classes is not None:
        include_names = {_normalize_label_name(name) for name in include_classes}

    ordered_category_names = [name for _, name in sorted(categories.items(), key=lambda x: x[0])]
    selected_label_names = [
        name for name in ordered_category_names if include_names is None or _normalize_label_name(name) in include_names
    ]
    if not selected_label_names:
        return [], {}

    label_map = {name: idx for idx, name in enumerate(selected_label_names)}
    label_caps = _build_per_label_caps(selected_label_names, max_samples_per_label)
    label_counts = {name: 0 for name in selected_label_names}

    records: list[dict[str, Any]] = []
    for image_id, image_info in images_by_id.items():
        subset = image_info.get("subset", "train")
        width = int(image_info["width"])
        height = int(image_info["height"])
        image_path = image_dirs[subset] / image_info["file_name"]

        for ann in annots_by_image.get(image_id, []):
            class_name = categories[int(ann["category_id"])]
            if class_name not in label_map:
                continue
            cap = label_caps[class_name]
            if cap is not None and label_counts[class_name] >= cap:
                continue

            tooth_box = _resolve_bbox(ann, width, height)
            if tooth_box is None:
                continue

            base_name = image_info["file_name"].split(".rf.")[0]
            records.append(
                {
                    "record_id": f"{image_id}_{ann['id']}",
                    "group_id": base_name,
                    "subset": subset,
                    "source_image_id": image_id,
                    "image_path": str(image_path),
                    "file_name": image_info["file_name"],
                    "crop_box": expand_box_xyxy(tooth_box, width, height, margin=crop_margin),
                    "tooth_box": tooth_box,
                    "label": label_map[class_name],
                    "label_name": class_name,
                    "source_class": class_name,
                }
            )
            label_counts[class_name] += 1

    return records, label_map


def build_classification_records_from_masks(
    coco_data: dict[str, Any],
    image_dirs: dict[str, Path],
    *,
    positive_classes: Iterable[str] = ("Caries",),
    crop_margin: float = 0.08,
) -> list[dict[str, Any]]:
    multiclass_records, _ = build_multiclass_classification_records_from_masks(
        coco_data,
        image_dirs,
        include_classes=positive_classes,
        crop_margin=crop_margin,
    )

    records: list[dict[str, Any]] = []
    for record in multiclass_records:
        binary_record = dict(record)
        binary_record["label"] = 1
        records.append(binary_record)
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


def summarize_multiclass_labels(records: list[dict[str, Any]], label_name_key: str = "label_name") -> dict[str, int]:
    summary: dict[str, int] = defaultdict(int)
    for record in records:
        label_name = str(record.get(label_name_key, "unknown"))
        summary[label_name] += 1
    return dict(sorted(summary.items(), key=lambda x: x[0].lower()))
