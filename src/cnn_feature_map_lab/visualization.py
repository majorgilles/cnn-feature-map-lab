"""Small visualization helpers for image tensors and feature maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image


def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Normalize one feature map to the 0..1 range for display."""
    feature = feature_map.detach().cpu().float()
    feature_min = feature.min()
    feature_max = feature.max()
    if torch.isclose(feature_min, feature_max):
        return torch.zeros_like(feature)
    return (feature - feature_min) / (feature_max - feature_min)


def save_tensor_image(tensor: torch.Tensor, output_path: Path, title: str | None = None) -> None:
    """Save a CHW or BCHW image tensor as a small figure."""
    image = tensor.detach().cpu()
    if image.ndim == 4:
        image = image[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(to_pil_image(image.clamp(0, 1)))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
