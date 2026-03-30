from __future__ import annotations

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from classification_pipeline import (
    ClassificationDownloadConfig,
    ToothCropDataset,
    build_classification_image_pipeline,
    build_classification_records_from_masks,
    build_classification_resize_pipeline,
    load_or_download_classification_dataset,
    split_grouped_records as split_classification_records,
    summarize_binary_labels,
)
from detection_pipeline import (
    AugmentedToothDetectionDataset,
    DetectionDownloadConfig,
    ToothDetectionDataset,
    build_detection_records,
    build_detection_train_pipeline,
    load_or_download_detection_dataset,
    split_grouped_records as split_detection_records,
)


def _to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.shape[-1] == 1:
        image_np = image_np[..., 0]
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np


def _sample_indices(dataset_len: int, sample_count: int) -> list[int]:
    count = min(sample_count, dataset_len)
    if count == 0:
        return []
    return random.sample(range(dataset_len), count)


def _show_detection_examples(name: str, dataset, sample_count: int = 20) -> None:
    indices = _sample_indices(len(dataset), sample_count)
    if not indices:
        print(f"No samples available for {name}.")
        return

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes_arr, indices):
        image_tensor, target = dataset[idx]
        image = _to_display_image(image_tensor)
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

        boxes = target["boxes"].detach().cpu().numpy()
        labels = target["labels"].detach().cpu().numpy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="lime")
            ax.add_patch(rect)
            ax.text(x1, max(0.0, y1 - 3), str(int(label)), color="yellow", fontsize=8, backgroundcolor="black")

        ax.set_title(f"idx={idx} boxes={len(boxes)}", fontsize=9)
        ax.axis("off")

    for ax in axes_arr[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"{name} - {len(indices)} examples", fontsize=14)
    fig.tight_layout()
    plt.show(block=False)


def _show_classification_examples(name: str, dataset, sample_count: int = 20) -> None:
    indices = _sample_indices(len(dataset), sample_count)
    if not indices:
        print(f"No samples available for {name}.")
        return

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes_arr, indices):
        image_tensor, label_tensor = dataset[idx]
        image = _to_display_image(image_tensor)
        label = int(label_tensor.item())
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(f"idx={idx} label={label}", fontsize=9)
        ax.axis("off")

    for ax in axes_arr[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"{name} - {len(indices)} examples", fontsize=14)
    fig.tight_layout()
    plt.show(block=False)


def main() -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    force_download = any(arg.lower() == "download" for arg in sys.argv[1:])

    if force_download:
        print("Forced download enabled via CLI argument: download")
    else:
        print("Using cached datasets when available. Add 'download' argument to force refresh.")

    detection_root, detection_coco, detection_image_dirs = load_or_download_detection_dataset(
        DetectionDownloadConfig(api_key=api_key),
        force_download=force_download,
    )
    detection_records, tooth_label_map = build_detection_records(detection_coco, detection_image_dirs)
    det_train, det_val, det_test = split_detection_records(detection_records)

    detection_train = AugmentedToothDetectionDataset(
        ToothDetectionDataset(det_train, image_size=640, output_channels=3),
        build_detection_train_pipeline(),
    )
    detection_val = ToothDetectionDataset(det_val, image_size=640, output_channels=3)
    detection_test = ToothDetectionDataset(det_test, image_size=640, output_channels=3)

    classification_root, classification_coco, classification_image_dirs = load_or_download_classification_dataset(
        ClassificationDownloadConfig(api_key=api_key),
        force_download=force_download,
    )
    classification_records = build_classification_records_from_masks(
        classification_coco,
        classification_image_dirs,
        positive_classes=("Caries",),
        crop_margin=0.08,
    )
    cls_train, cls_val, cls_test = split_classification_records(classification_records)

    resize = build_classification_resize_pipeline(224)
    classification_train = ToothCropDataset(
        cls_train,
        image_size=224,
        output_channels=1,
        resize_transform=resize,
        image_transform=build_classification_image_pipeline(),
    )
    classification_val = ToothCropDataset(
        cls_val,
        image_size=224,
        output_channels=1,
        resize_transform=resize,
    )
    classification_test = ToothCropDataset(
        cls_test,
        image_size=224,
        output_channels=1,
        resize_transform=resize,
    )

    print(f"Detection dataset root: {detection_root}")
    print(f"Tooth classes: {len(tooth_label_map)}")
    print(f"Detection train/val/test: {len(detection_train)}, {len(detection_val)}, {len(detection_test)}")

    print(f"Classification dataset root: {classification_root}")
    print("Classification records are positive carious-tooth crops derived from segmentation masks.")
    print(f"Classification train summary: {summarize_binary_labels(cls_train)}")
    print(f"Classification train/val/test: {len(classification_train)}, {len(classification_val)}, {len(classification_test)}")

    _show_detection_examples("Detection train (augmented)", detection_train, sample_count=20)
    _show_detection_examples("Detection val", detection_val, sample_count=20)
    _show_detection_examples("Detection test", detection_test, sample_count=20)

    _show_classification_examples("Classification train (augmented)", classification_train, sample_count=20)
    _show_classification_examples("Classification val", classification_val, sample_count=20)
    _show_classification_examples("Classification test", classification_test, sample_count=20)

    print("Close all figure windows to finish.")
    plt.show()


if __name__ == "__main__":
    main()
