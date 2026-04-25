from __future__ import annotations

import albumentations as A
import cv2

DETECTION_NOISE_STD_RANGE = (0.01, 0.04)


def build_detection_train_pipeline() -> A.Compose:
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
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.20),
            A.GaussNoise(std_range=DETECTION_NOISE_STD_RANGE, mean_range=(0.0, 0.0), p=0.15),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.20),
    )
