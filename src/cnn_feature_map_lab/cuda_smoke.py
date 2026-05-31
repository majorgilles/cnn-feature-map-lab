"""Tiny CUDA smoke test for the lab environment."""

import torch

from cnn_feature_map_lab.device import describe_device


def main() -> None:
    """Print the selected device and run one tensor operation there."""
    info = describe_device()
    print(f"Selected device: {info.device}")
    print(f"CUDA available: {info.cuda_available}")
    if info.cuda_device_name is not None:
        print(f"CUDA device name: {info.cuda_device_name}")

    torch.tensor([1.0, 2.0, 3.0], device=info.device)
