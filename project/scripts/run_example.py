from __future__ import annotations

import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from utils.data_split import split_records_by_subset
from utils.display_item import to_display_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from classification_pipeline import (
    ClassificationDownloadConfig,
    ToothCropDataset,
    build_classification_image_pipeline,
    build_multiclass_classification_records_from_masks,
    build_classification_resize_pipeline,
    load_or_download_classification_dataset,
    summarize_multiclass_labels,
)
from detection_pipeline import (
    AugmentedToothDetectionDataset,
    DetectionDownloadConfig,
    ToothDetectionDataset,
    build_detection_records,
    build_detection_train_pipeline,
    load_or_download_detection_dataset,
    split_grouped_records
)


def _sample_indices(dataset_len: int, sample_count: int) -> list[int]:
    count = min(sample_count, dataset_len)
    if count == 0:
        return []
    return random.sample(range(dataset_len), count)


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _build_output_paths(group_root: str, name: str) -> tuple[str, str]:
    os.makedirs(group_root, exist_ok=True)
    clean_name = _safe_name(name)
    txt_filepath = os.path.join(group_root, f"{clean_name}_file_list.txt")
    img_filepath = os.path.join(group_root, f"{clean_name}.jpg")
    return txt_filepath, img_filepath


def _show_detection_examples(name: str, dataset, output_dir: str, sample_count: int = 20) -> None:
    indices = _sample_indices(len(dataset), sample_count)
    if not indices:
        print(f"No samples available for {name}.")
        return

    txt_filepath, img_filepath = _build_output_paths(output_dir, name)

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(f"--- File list for: {name} ---\n")
        f.write("Format: [Index in Plot] - [Original Filename]\n\n")

        for ax, idx in zip(axes_arr, indices):
            image_tensor, target = dataset[idx]
            image = to_display_image(image_tensor)
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

            file_name = target.get("file_name", "unknown")
            f.write(f"{idx} - {file_name}\n")

            boxes = target["boxes"].detach().cpu().numpy()
            labels = target["labels"].detach().cpu().numpy()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=1.5,
                    edgecolor="lime",
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    max(0.0, y1 - 3),
                    str(int(label)),
                    color="yellow",
                    fontsize=8,
                    backgroundcolor="black",
                )

            ax.set_title(f"idx={idx}", fontsize=10, fontweight="bold")
            ax.axis("off")

    for ax in axes_arr[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"{name} - {len(indices)} examples", fontsize=14)
    fig.tight_layout()
    plt.savefig(img_filepath)
    plt.close(fig)


def _show_classification_examples(name: str, dataset, output_dir: str, sample_count: int = 20) -> None:
    indices = _sample_indices(len(dataset), sample_count)
    if not indices:
        print(f"No samples available for {name}.")
        return

    txt_filepath, img_filepath = _build_output_paths(output_dir, name)

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(f"--- File list for: {name} ---\n")
        f.write("Format: [Index in Plot] - [Original Filename] - [Label Name]\n\n")

        for ax, idx in zip(axes_arr, indices):
            image_tensor, label_tensor, metadata = dataset[idx]

            image = to_display_image(image_tensor)
            label = int(label_tensor.item())
            label_name = metadata.get("label_name", metadata.get("source_class", "unknown"))
            file_name = metadata.get("file_name", "unknown")

            f.write(f"{idx} - {file_name} - {label_name}\n")

            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(f"idx={idx} L={label} ({label_name})", fontsize=10, fontweight="bold")
            ax.axis("off")

    for ax in axes_arr[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"{name} - {len(indices)} examples", fontsize=14)
    fig.tight_layout()
    plt.savefig(img_filepath)
    plt.close(fig)


def _show_filtered_classification_examples(
    name: str,
    dataset,
    output_dir: str,
    sample_count: int,
    predicate,
    empty_message: str,
) -> None:
    filtered_indices = [
        idx
        for idx, record in enumerate(dataset.records)
        if predicate(str(record.get("label_name", record.get("source_class", ""))))
    ]

    if not filtered_indices:
        print(empty_message)
        return

    count = min(sample_count, len(filtered_indices))
    indices = random.sample(filtered_indices, count)

    txt_filepath, img_filepath = _build_output_paths(output_dir, name)

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(f"--- File list for: {name} ---\n")
        f.write("Format: [Index in Plot] - [Original Filename] - [Label Name]\n\n")

        for ax, idx in zip(axes_arr, indices):
            image_tensor, label_tensor, metadata = dataset[idx]
            image = to_display_image(image_tensor)
            label = int(label_tensor.item())
            label_name = metadata.get("label_name", metadata.get("source_class", "unknown"))
            file_name = metadata.get("file_name", "unknown")

            f.write(f"{idx} - {file_name} - {label_name}\n")

            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(f"idx={idx} L={label} ({label_name})", fontsize=10, fontweight="bold")
            ax.axis("off")

    for ax in axes_arr[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"{name} - {len(indices)} examples", fontsize=14)
    fig.tight_layout()
    plt.savefig(img_filepath)
    plt.close(fig)


def _show_caries_classification_examples(name: str, dataset, output_dir: str, sample_count: int = 20) -> None:
    _show_filtered_classification_examples(
        name=name,
        dataset=dataset,
        output_dir=output_dir,
        sample_count=sample_count,
        predicate=lambda label: label.lower() == "caries",
        empty_message=f"No caries-only samples available for {name}.",
    )


def _show_non_caries_classification_examples(name: str, dataset, output_dir: str, sample_count: int = 20) -> None:
    _show_filtered_classification_examples(
        name=name,
        dataset=dataset,
        output_dir=output_dir,
        sample_count=sample_count,
        predicate=lambda label: label.lower() != "caries",
        empty_message=f"No non-caries samples available for {name}.",
    )


def _parse_max_samples_per_label(args: list[str]) -> int | dict[str, int] | None:
    """
    Supports:
    - max_per_label=300
    - max_per_label=Caries:1000,Implant:300,Filling:300
    """
    token = next((arg for arg in args if arg.startswith("max_per_label=")), None)
    if token is None:
        return None

    value = token.split("=", 1)[1].strip()
    if not value:
        raise ValueError("max_per_label value is empty")

    if ":" not in value:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("max_per_label must be > 0")
        return parsed

    per_class: dict[str, int] = {}
    for pair in value.split(","):
        key, raw_count = pair.split(":", 1)
        key = key.strip()
        count = int(raw_count.strip())
        if not key:
            raise ValueError(f"Invalid class key in max_per_label: '{pair}'")
        if count <= 0:
            raise ValueError(f"max_per_label for '{key}' must be > 0")
        per_class[key] = count
    return per_class


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    detection_output_dir = os.path.join(config.OUTPUT_DIR, "detection")
    classification_output_dir = os.path.join(config.OUTPUT_DIR, "classification")
    classification_mixed_output_dir = os.path.join(classification_output_dir, "mixed")
    classification_caries_output_dir = os.path.join(classification_output_dir, "caries")
    classification_non_caries_output_dir = os.path.join(classification_output_dir, "non_caries")

    os.makedirs(detection_output_dir, exist_ok=True)
    os.makedirs(classification_mixed_output_dir, exist_ok=True)
    os.makedirs(classification_caries_output_dir, exist_ok=True)
    os.makedirs(classification_non_caries_output_dir, exist_ok=True)

    api_key = os.getenv("ROBOFLOW_API_KEY")
    cli_args = sys.argv[1:]
    force_download = any(arg.lower() == "download" for arg in cli_args)
    max_samples_per_label = _parse_max_samples_per_label(cli_args)

    if force_download:
        print("Forced download enabled via CLI argument: download")
    else:
        print("Using cached datasets when available. Add 'download' argument to force refresh.")

    detection_root, detection_coco, detection_image_dirs = load_or_download_detection_dataset(
        DetectionDownloadConfig(api_key=api_key),
        force_download=force_download,
    )
    detection_records, tooth_label_map = build_detection_records(detection_coco, detection_image_dirs)
    det_train, det_val, det_test = split_grouped_records(
        detection_records,
        train_size=config.TRAIN_RATIO,
        val_size=config.VAL_RATIO,
        test_size=config.TEST_RATIO,
        random_state=42
    )

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
    classification_records, classification_label_map = build_multiclass_classification_records_from_masks(
        classification_coco,
        classification_image_dirs,
        crop_margin=0.08,
        max_samples_per_label=max_samples_per_label,
    )
    cls_train, cls_val, cls_test = split_records_by_subset(classification_records)

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
    print("Classification records are tooth crops derived from segmentation masks across all available labels.")
    print(f"Classification label map: {classification_label_map}")
    print(f"Classification train summary: {summarize_multiclass_labels(cls_train)}")
    print(f"Classification train/val/test: {len(classification_train)}, {len(classification_val)}, {len(classification_test)}")

    mixed_sample_count = 20

    _show_detection_examples(
        "Detection train (augmented)",
        detection_train,
        output_dir=detection_output_dir,
        sample_count=20,
    )
    _show_detection_examples(
        "Detection val",
        detection_val,
        output_dir=detection_output_dir,
        sample_count=20,
    )
    _show_detection_examples(
        "Detection test",
        detection_test,
        output_dir=detection_output_dir,
        sample_count=20,
    )

    _show_classification_examples(
        "Classification train (augmented)",
        classification_train,
        output_dir=classification_mixed_output_dir,
        sample_count=mixed_sample_count,
    )
    _show_classification_examples(
        "Classification val",
        classification_val,
        output_dir=classification_mixed_output_dir,
        sample_count=20,
    )
    _show_classification_examples(
        "Classification test",
        classification_test,
        output_dir=classification_mixed_output_dir,
        sample_count=mixed_sample_count,
    )

    _show_caries_classification_examples(
        "Classification train (augmented) caries-only",
        classification_train,
        output_dir=classification_caries_output_dir,
        sample_count=mixed_sample_count,
    )
    _show_caries_classification_examples(
        "Classification test caries-only",
        classification_test,
        output_dir=classification_caries_output_dir,
        sample_count=mixed_sample_count,
    )

    _show_non_caries_classification_examples(
        "Classification train non-caries",
        classification_train,
        output_dir=classification_non_caries_output_dir,
        sample_count=20,
    )

    print("Example images saved to output directory.")
    print(f"Detection examples: {detection_output_dir}")
    print(f"Classification mixed examples: {classification_mixed_output_dir}")
    print(f"Classification caries examples: {classification_caries_output_dir}")
    print(f"Classification non-caries examples: {classification_non_caries_output_dir}")

    plt.show()


if __name__ == "__main__":
    main()