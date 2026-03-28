from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A

from ..preprocessing import load_radiograph, to_chw_tensor


class ToothDetectionDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        *,
        image_size: int = 640,
        output_channels: int = 3,
        denoise: bool = True,
    ) -> None:
        self.records = records
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

    def get_raw_sample(self, index: int) -> dict:
        record = self.records[index]
        image = load_radiograph(
            record["image_path"],
            denoise=self.denoise,
            output_channels=self.output_channels,
        )
        transformed = self.resize(image=image, bboxes=record["boxes"], class_labels=record["labels"])
        return {
            "image_np": transformed["image"].astype(np.float32),
            "boxes": np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(transformed["class_labels"], dtype=np.int64),
            "image_id": int(record["image_id"]),
        }

    def __getitem__(self, index: int):
        sample = self.get_raw_sample(index)
        boxes = sample["boxes"]
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else np.zeros((0,), dtype=np.float32)
        )
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(sample["labels"]),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
            "area": torch.from_numpy(area.astype(np.float32)),
            "iscrowd": torch.zeros((len(sample["labels"]),), dtype=torch.int64),
        }
        return to_chw_tensor(sample["image_np"]), target


class AugmentedToothDetectionDataset(Dataset):
    def __init__(self, base_dataset: ToothDetectionDataset, transform: A.Compose) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset.get_raw_sample(index)
        transformed = self.transform(
            image=sample["image_np"],
            bboxes=sample["boxes"].tolist(),
            class_labels=sample["labels"].tolist(),
        )
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
        return to_chw_tensor(transformed["image"]), target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
