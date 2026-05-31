"""Device helpers for PyTorch experiments."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Small summary of the selected PyTorch device."""

    device: torch.device
    cuda_available: bool
    cuda_device_name: str | None


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when available and requested; otherwise return CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> DeviceInfo:
    """Return a typed summary of the selected device."""
    selected_device = device or get_device()
    cuda_device_name = None
    if selected_device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(selected_device)
    return DeviceInfo(
        device=selected_device,
        cuda_available=torch.cuda.is_available(),
        cuda_device_name=cuda_device_name,
    )
