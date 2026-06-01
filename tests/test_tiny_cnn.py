import torch

from cnn_feature_map_lab.tiny_cnn import TinyCifar10Cnn


def test_tiny_cifar10_cnn_output_shape() -> None:
    model = TinyCifar10Cnn()
    images = torch.zeros(2, 3, 32, 32)

    outputs = model(images)

    assert outputs.shape == (2, 10)
