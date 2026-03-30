from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def robust_minmax_normalize(image: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0) -> np.ndarray:
    image = image.astype(np.float32)
    lo = np.percentile(image, lower_percentile)
    hi = np.percentile(image, upper_percentile)
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return ((image - lo) / (hi - lo)).astype(np.float32)


def load_radiograph(image_path: str | Path, *, denoise: bool = True, output_channels: int = 1) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = robust_minmax_normalize(image)
    if denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    image = image[..., None]
    if output_channels == 3:
        image = np.repeat(image, 3, axis=2)
    elif output_channels != 1:
        raise ValueError("output_channels must be 1 or 3")
    return image.astype(np.float32)


def to_chw_tensor(image: np.ndarray):
    import torch

    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
