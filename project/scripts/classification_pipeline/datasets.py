from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_radiograph, to_chw_tensor


def _safe_crop(image: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for box {box}")
    return crop


class ToothCropDataset(Dataset):
    def __init__(self, records: list[dict], *, image_size: int = 224, output_channels: int = 1, denoise: bool = True, resize_transform=None, image_transform=None) -> None:
        self.records = records
        self.output_channels = output_channels
        self.denoise = denoise
        self.resize_transform = resize_transform
        self.image_transform = image_transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def get_raw_sample(self, index: int) -> dict:
        record = self.records[index]
        image = load_radiograph(record["image_path"], denoise=self.denoise, output_channels=1)
        crop_box = record["crop_box"]

        if self.image_transform is not None:
            transformed = self.image_transform(image=image, bboxes=[crop_box], class_labels=[1])
            image = transformed["image"]
            crop_box = transformed["bboxes"][0]

        crop = _safe_crop(image, crop_box)
        if self.resize_transform is not None:
            crop = self.resize_transform(image=crop)["image"]

        if self.output_channels == 3 and crop.shape[-1] == 1:
            crop = np.repeat(crop, 3, axis=2)
        elif self.output_channels == 1 and crop.ndim == 2:
            crop = crop[..., None]

        return {
            "image_np": crop.astype(np.float32),
            "label": int(record["label"]),
            "record_id": record["record_id"],
            "source_image_id": int(record["source_image_id"]),
            "source_class": record["source_class"],
        }

    def __getitem__(self, index: int):
        sample = self.get_raw_sample(index)
        return to_chw_tensor(sample["image_np"]), torch.tensor(sample["label"], dtype=torch.long)
