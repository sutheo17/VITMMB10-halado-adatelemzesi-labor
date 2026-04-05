import torch
import numpy as np

def to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.shape[-1] == 1:
        image_np = image_np[..., 0]
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np
