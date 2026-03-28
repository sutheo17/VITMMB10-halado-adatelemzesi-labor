
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from roboflow import Roboflow
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
except ImportError:
    L = None


@dataclass
class RoboflowConfig:
    workspace: str = "mohamed-uob"
    project_name: str = "denim"
    version: int = 1
    export_format: str = "coco"
    api_key: Optional[str] = None

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not key:
            raise ValueError(
                "Roboflow API key not found. "
                "Set the ROBOFLOW_API_KEY environment variable or pass api_key explicitly."
            )
        return key


def download_roboflow_coco(config: RoboflowConfig) -> tuple[Path, Path, dict[str, Any]]:
    """
    Download a Roboflow dataset in COCO format and return:
      - dataset root
      - directory where the images and annotations live
      - loaded COCO json
    """
    rf = Roboflow(api_key=config.resolved_api_key())
    project = rf.workspace(config.workspace).project(config.project_name)
    dataset = project.version(config.version).download(config.export_format)

    dataset_root = Path(dataset.location)
    json_files = sorted(dataset_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No COCO json file found under: {dataset_root}")

    json_path = json_files[0]
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    return dataset_root, json_path.parent, coco_data


def build_annotation_index(coco_data: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
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
        cid
        for cid, name in categories.items()
        if name.isdigit() and 1 <= int(name) <= 32
    )
    if not tooth_ids:
        raise ValueError("No tooth categories 1..32 were found in the COCO categories.")

    caries_id = next((cid for cid, name in categories.items() if "caries" in name.lower()), None)
    if caries_id is None:
        raise ValueError("No 'caries' category found in the COCO categories.")

    tooth_label_map = {category_id: idx + 1 for idx, category_id in enumerate(tooth_ids)}
    return categories, tooth_ids, caries_id, tooth_label_map


def coco_bbox_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def clip_box_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def is_valid_xyxy(box: list[float], min_size: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def expand_box_xyxy(
    box: list[float],
    width: int,
    height: int,
    margin: float = 0.08,
) -> list[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(round(x1 - bw * margin)))
    y1 = max(0, int(round(y1 - bh * margin)))
    x2 = min(width, int(round(x2 + bw * margin)))
    y2 = min(height, int(round(y2 + bh * margin)))
    return [x1, y1, x2, y2]


def bbox_intersection_area(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def tooth_has_caries(
    tooth_box: list[float],
    caries_boxes: list[list[float]],
    min_intersection_ratio: float = 0.03,
) -> bool:
    """
    Heuristic for module 2:
    mark a tooth as carious if at least one caries box overlaps the tooth box
    or the center of the caries box falls inside the tooth box.
    """
    tooth_area = max(1.0, (tooth_box[2] - tooth_box[0]) * (tooth_box[3] - tooth_box[1]))

    for caries_box in caries_boxes:
        if point_in_box(box_center(caries_box), tooth_box):
            return True
        inter_area = bbox_intersection_area(tooth_box, caries_box)
        if inter_area / tooth_area >= min_intersection_ratio:
            return True
    return False


def robust_minmax_normalize(
    gray_image: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> np.ndarray:
    """
    Robust min-max normalization using percentiles.
    Output range: [0, 1].
    """
    gray = gray_image.astype(np.float32)
    lo = np.percentile(gray, lower_percentile)
    hi = np.percentile(gray, upper_percentile)

    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)

    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo)
    return gray.astype(np.float32)


def preprocess_radiograph(
    image_path: str | Path,
    *,
    denoise: bool = True,
    output_channels: int = 3,
) -> np.ndarray:
    """
    Load as grayscale -> robust normalize -> optional mild Gaussian denoise.
    Returns HWC float32 image in [0, 1].
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = robust_minmax_normalize(image)

    if denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    image = image[..., None]  # H, W, 1
    if output_channels == 3:
        image = np.repeat(image, 3, axis=2)
    elif output_channels != 1:
        raise ValueError("output_channels must be 1 or 3.")

    return image.astype(np.float32)


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
            if not is_valid_xyxy(box):
                continue

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


def build_module2_classification_records(
    coco_data: dict[str, Any],
    raw_dir: str | Path,
    *,
    crop_margin: float = 0.08,
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    images_by_id, annots_by_image = build_annotation_index(coco_data)
    categories, tooth_ids, caries_id, _ = build_category_metadata(coco_data)

    records: list[dict[str, Any]] = []

    for image_id, img_info in images_by_id.items():
        width = int(img_info["width"])
        height = int(img_info["height"])
        annots = annots_by_image.get(image_id, [])

        tooth_annots = [ann for ann in annots if int(ann["category_id"]) in tooth_ids]
        caries_boxes = [
            clip_box_xyxy(coco_bbox_to_xyxy(ann["bbox"]), width, height)
            for ann in annots
            if int(ann["category_id"]) == caries_id
        ]

        for tooth_ann in tooth_annots:
            tooth_category_id = int(tooth_ann["category_id"])
            tooth_box = clip_box_xyxy(coco_bbox_to_xyxy(tooth_ann["bbox"]), width, height)

            if not is_valid_xyxy(tooth_box):
                continue

            label = int(tooth_has_caries(tooth_box, caries_boxes))
            crop_box = expand_box_xyxy(tooth_box, width, height, margin=crop_margin)

            records.append(
                {
                    "record_id": f"{image_id}_{tooth_ann['id']}",
                    "group_id": image_id,
                    "source_image_id": image_id,
                    "image_path": str(raw_dir / img_info["file_name"]),
                    "file_name": img_info["file_name"],
                    "crop_box": crop_box,
                    "tooth_box": tooth_box,
                    "tooth_label_name": categories[tooth_category_id],
                    "label": label,
                }
            )

    return records


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

    all_groups = list(groups_to_records.keys())
    if len(all_groups) < 3:
        raise ValueError("Need at least 3 groups to build train/val/test splits.")

    def group_strat_labels(groups: list[Any]) -> Optional[list[int]]:
        if stratify_key is None:
            return None
        labels = [max(int(rec[stratify_key]) for rec in groups_to_records[group]) for group in groups]
        if len(set(labels)) < 2:
            return None
        return labels

    train_val_groups, test_groups = train_test_split(
        all_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=group_strat_labels(all_groups),
    )

    relative_val_ratio = val_size / (train_size + val_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=group_strat_labels(train_val_groups),
    )

    def flatten(groups: list[Any]) -> list[dict[str, Any]]:
        return [record for group in groups for record in groups_to_records[group]]

    return flatten(train_groups), flatten(val_groups), flatten(test_groups)


def build_detection_train_augmenter() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-7, 7),
                shear=(-2, 2),
                p=0.70,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.20,
            ),
            A.GaussNoise(var_limit=(3.0, 15.0), p=0.15),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.20,
        ),
    )


def build_classification_train_augmenter() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-10, 10),
                shear=(-2, 2),
                p=0.70,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.20,
            ),
            A.GaussNoise(var_limit=(3.0, 15.0), p=0.15),
        ]
    )


class ToothDetectionDataset(Dataset):
    """
    Base detection dataset with deterministic preprocessing only.
    No online augmentation happens here.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_size: int = 640,
        output_channels: int = 3,
        denoise: bool = True,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.output_channels = output_channels
        self.denoise = denoise
        self.resize = A.Compose(
            [A.Resize(image_size, image_size)],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,
            ),
        )

    def __len__(self) -> int:
        return len(self.records)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.astype(np.float32))

    def get_raw_sample(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = preprocess_radiograph(
            record["image_path"],
            denoise=self.denoise,
            output_channels=self.output_channels,
        )

        transformed = self.resize(
            image=image,
            bboxes=record["boxes"],
            class_labels=record["labels"],
        )

        boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(transformed["class_labels"], dtype=np.int64)

        return {
            "image_np": transformed["image"].astype(np.float32),
            "boxes": boxes,
            "labels": labels,
            "image_id": int(record["image_id"]),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.get_raw_sample(index)
        image_tensor = self._image_to_tensor(sample["image_np"])

        boxes = sample["boxes"]
        labels = sample["labels"]
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else np.zeros((0,), dtype=np.float32)
        )

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
            "area": torch.from_numpy(area.astype(np.float32)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


class AugmentedToothDetectionDataset(Dataset):
    """
    Wrapper dataset: performs online train-time augmentation on each access.
    """

    def __init__(self, base_dataset: ToothDetectionDataset, transform: A.Compose) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.base_dataset.get_raw_sample(index)

        transformed = self.transform(
            image=sample["image_np"],
            bboxes=sample["boxes"].tolist(),
            class_labels=sample["labels"].tolist(),
        )

        image_tensor = self.base_dataset._image_to_tensor(transformed["image"])
        boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(transformed["class_labels"], dtype=np.int64)

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else np.zeros((0,), dtype=np.float32)
        )

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
            "area": torch.from_numpy(area.astype(np.float32)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


class ToothCropDataset(Dataset):
    """
    Base classification dataset for tooth crops.
    Deterministic preprocessing only.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_size: int = 224,
        output_channels: int = 1,
        denoise: bool = True,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.output_channels = output_channels
        self.denoise = denoise
        self.resize = A.Compose([A.Resize(image_size, image_size)])

    def __len__(self) -> int:
        return len(self.records)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.astype(np.float32))

    def get_raw_sample(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        base_image = preprocess_radiograph(
            record["image_path"],
            denoise=self.denoise,
            output_channels=1,
        )

        x1, y1, x2, y2 = record["crop_box"]
        crop = base_image[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError(f"Empty crop for record {record['record_id']}")

        crop = self.resize(image=crop)["image"]

        if self.output_channels == 3 and crop.shape[-1] == 1:
            crop = np.repeat(crop, 3, axis=2)
        elif self.output_channels == 1 and crop.ndim == 2:
            crop = crop[..., None]

        return {
            "image_np": crop.astype(np.float32),
            "label": int(record["label"]),
            "record_id": record["record_id"],
            "source_image_id": int(record["source_image_id"]),
            "tooth_label_name": record["tooth_label_name"],
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.get_raw_sample(index)
        image_tensor = self._image_to_tensor(sample["image_np"])
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return image_tensor, label_tensor


class AugmentedToothCropDataset(Dataset):
    """
    Wrapper dataset: performs online train-time augmentation on each access.
    """

    def __init__(self, base_dataset: ToothCropDataset, transform: A.Compose) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.base_dataset.get_raw_sample(index)
        transformed = self.transform(image=sample["image_np"])
        image_tensor = self.base_dataset._image_to_tensor(transformed["image"])
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return image_tensor, label_tensor


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


class ToothCariesDataModule(L.LightningDataModule if L is not None else object):
    """
    Lightning-compatible DataModule for module 2 classification.
    Create the record splits first, then pass them in here.
    """

    def __init__(
        self,
        train_records: list[dict[str, Any]],
        val_records: list[dict[str, Any]],
        test_records: list[dict[str, Any]],
        *,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        output_channels: int = 1,
    ) -> None:
        if L is None:
            raise ImportError("lightning is not installed. Install it to use ToothCariesDataModule.")

        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_channels = output_channels

    def setup(self, stage: Optional[str] = None) -> None:
        train_base = ToothCropDataset(
            self.train_records,
            image_size=self.image_size,
            output_channels=self.output_channels,
        )
        self.train_dataset = AugmentedToothCropDataset(
            train_base,
            build_classification_train_augmenter(),
        )

        self.val_dataset = ToothCropDataset(
            self.val_records,
            image_size=self.image_size,
            output_channels=self.output_channels,
        )
        self.test_dataset = ToothCropDataset(
            self.test_records,
            image_size=self.image_size,
            output_channels=self.output_channels,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def visualize_detection_samples(
    dataset: Dataset,
    *,
    num_samples: int = 4,
    figsize: tuple[int, int] = (16, 4),
) -> None:
    plt.figure(figsize=figsize)

    for plot_idx in range(num_samples):
        image_tensor, target = dataset[plot_idx]
        image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

        if image.shape[-1] == 1:
            image = image[..., 0]
            plt.subplot(1, num_samples, plot_idx + 1)
            plt.imshow(image, cmap="gray")
        else:
            plt.subplot(1, num_samples, plot_idx + 1)
            plt.imshow(image)

        for box in target["boxes"].cpu().numpy():
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.5,
            )
            plt.gca().add_patch(rect)

        plt.axis("off")
        plt.title(f"Detection sample {plot_idx + 1}")

    plt.tight_layout()
    plt.show()


def visualize_classification_samples(
    dataset: Dataset,
    *,
    num_samples: int = 8,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    rows = int(np.ceil(num_samples / 4))
    plt.figure(figsize=figsize)

    for plot_idx in range(num_samples):
        image_tensor, label_tensor = dataset[plot_idx]
        image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

        plt.subplot(rows, 4, plot_idx + 1)
        if image.shape[-1] == 1:
            plt.imshow(image[..., 0], cmap="gray")
        else:
            plt.imshow(image)
        plt.title(f"label={int(label_tensor.item())}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def summarize_binary_labels(records: list[dict[str, Any]], label_key: str = "label") -> dict[str, int]:
    labels = [int(record[label_key]) for record in records]
    positives = int(sum(labels))
    negatives = int(len(labels) - positives)
    return {"total": len(labels), "positive": positives, "negative": negatives}


if __name__ == "__main__":
    config = RoboflowConfig(
        workspace="mohamed-uob",
        project_name="denim",
        version=1,
        api_key=os.getenv("ROBOFLOW_API_KEY"),
    )

    dataset_root, raw_dir, coco_data = download_roboflow_coco(config)

    module1_records, tooth_label_map = build_module1_detection_records(coco_data, raw_dir)
    module2_records = build_module2_classification_records(coco_data, raw_dir)

    module1_train, module1_val, module1_test = split_grouped_records(
        module1_records,
        group_key="group_id",
        stratify_key=None,
    )
    module2_train, module2_val, module2_test = split_grouped_records(
        module2_records,
        group_key="group_id",
        stratify_key="label",
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Module 1 detection records: {len(module1_records)}")
    print(f"Module 2 classification records: {len(module2_records)}")
    print(f"Module 2 train summary: {summarize_binary_labels(module2_train)}")
    print(f"Module 2 val summary: {summarize_binary_labels(module2_val)}")
    print(f"Module 2 test summary: {summarize_binary_labels(module2_test)}")

    detection_train_base = ToothDetectionDataset(module1_train, image_size=640, output_channels=3)
    detection_train_dataset = AugmentedToothDetectionDataset(
        detection_train_base,
        build_detection_train_augmenter(),
    )
    detection_val_dataset = ToothDetectionDataset(module1_val, image_size=640, output_channels=3)
    detection_test_dataset = ToothDetectionDataset(module1_test, image_size=640, output_channels=3)

    classification_train_base = ToothCropDataset(module2_train, image_size=224, output_channels=1)
    classification_train_dataset = AugmentedToothCropDataset(
        classification_train_base,
        build_classification_train_augmenter(),
    )
    classification_val_dataset = ToothCropDataset(module2_val, image_size=224, output_channels=1)
    classification_test_dataset = ToothCropDataset(module2_test, image_size=224, output_channels=1)

    print(f"Detection train/val/test: {len(detection_train_dataset)}, {len(detection_val_dataset)}, {len(detection_test_dataset)}")
    print(f"Classification train/val/test: {len(classification_train_dataset)}, {len(classification_val_dataset)}, {len(classification_test_dataset)}")

    visualize_detection_samples(detection_train_dataset, num_samples=4)
    visualize_classification_samples(classification_train_dataset, num_samples=8)
