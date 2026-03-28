from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np

from dental_data.download import RoboflowConfig, download_roboflow_coco
from dental_data.records import (
    build_module1_detection_records,
    split_grouped_records,
    summarize_binary_labels,
)
from dental_data.datasets.detection import (
    AugmentedToothDetectionDataset,
    ToothDetectionDataset,
)
from dental_data.pipelines.detection import build_detection_train_pipeline


def _tensor_image_to_uint8(image) -> np.ndarray:
    image_np = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.ndim == 2:
        image_np = image_np[..., None]
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    image_np = np.clip(image_np, 0.0, 1.0)
    return (image_np * 255.0).astype(np.uint8)


def _draw_boxes(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    vis = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=(0, 220, 0), thickness=2)
        cv2.putText(
            vis,
            str(int(label)),
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def save_detection_samples(
    split_name: str,
    dataset,
    output_dir: str | Path,
    max_samples: int = 3,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    count = min(max_samples, len(dataset))
    for index in range(count):
        image, target = dataset[index]
        image_uint8 = _tensor_image_to_uint8(image)
        boxes = target["boxes"].detach().cpu().numpy()
        labels = target["labels"].detach().cpu().numpy()
        vis = _draw_boxes(image_uint8, boxes, labels)
        save_path = output_path / f"{split_name}_{index:03d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    config = RoboflowConfig(api_key=os.getenv("ROBOFLOW_API_KEY"))
    dataset_root, raw_dir, coco_data = download_roboflow_coco(config)

    module1_records, _ = build_module1_detection_records(coco_data, raw_dir)

    module1_train, module1_val, module1_test = split_grouped_records(module1_records)

    detection_train = AugmentedToothDetectionDataset(
        ToothDetectionDataset(module1_train, image_size=640, output_channels=3),
        build_detection_train_pipeline(),
    )
    detection_val = ToothDetectionDataset(module1_val, image_size=640, output_channels=3)
    detection_test = ToothDetectionDataset(module1_test, image_size=640, output_channels=3)

    print(f"Dataset root: {dataset_root}")
    print(f"Detection train/val/test: {len(detection_train)}, {len(detection_val)}, {len(detection_test)}")

    preview_dir = Path(dataset_root) / "previews"
    save_detection_samples("train", detection_train, preview_dir, max_samples=3)
    save_detection_samples("val", detection_val, preview_dir, max_samples=3)
    save_detection_samples("test", detection_test, preview_dir, max_samples=3)
    print(f"Saved dataset previews to: {preview_dir}")


if __name__ == "__main__":
    main()
