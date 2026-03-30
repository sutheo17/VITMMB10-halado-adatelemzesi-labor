from __future__ import annotations

import albumentations as A
import cv2

CLASSIFICATION_NOISE_STD_RANGE = (0.005, 0.03)


def build_classification_image_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-7, 7),
                shear={"x": (-2, 2), "y": (-2, 2)},
                interpolation=cv2.INTER_LINEAR,
                fit_output=False,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.70,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.15),
            A.GaussNoise(std_range=CLASSIFICATION_NOISE_STD_RANGE, mean_range=(0.0, 0.0), p=0.12),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.95),
    )


def build_classification_resize_pipeline(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR)])
