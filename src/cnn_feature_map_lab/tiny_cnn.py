"""Tiny CNN architecture shared by the CIFAR-10 lab notebooks.

The helper keeps one source of truth for the small model used in Day 3 training and Day 4
feature-map inspection. A checkpoint can only load cleanly when this architecture has the same
layer names and tensor shapes that were used when the checkpoint was saved.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TinyCifar10Cnn(nn.Module):
    """Small CNN from the official PyTorch CIFAR-10 tutorial.

    The model is intentionally tiny because classification is only a vehicle for this lab. The
    real learning target is to create convolution filters and feature maps that are small enough to
    inspect visually. Modest CIFAR-10 accuracy is expected: the images are only 32 by 32 pixels, the
    model has very little capacity compared with modern CNNs, and the training run is deliberately
    short.

    Convolution layers:
        ``conv1`` accepts RGB images with 3 input channels and creates 6 feature maps using 5 by 5
        filters. ``conv2`` accepts those 6 feature maps and creates 16 new feature maps. The first
        convolution can learn simple local patterns such as color contrasts, edges, or corners. The
        second convolution combines the first layer's responses into slightly richer patterns.

    Pooling and shape changes:
        ``pool`` is a 2 by 2 max-pooling layer. It shrinks each feature map by keeping the strongest
        response in each 2 by 2 window. For one CIFAR-10 image, the shape path is
        ``3 x 32 x 32 -> 6 x 28 x 28 -> 6 x 14 x 14 -> 16 x 10 x 10 -> 16 x 5 x 5``.

    Fully connected layers:
        After the second pooling step, ``torch.flatten(x, 1)`` turns ``16 x 5 x 5`` feature values
        into 400 numbers per image. The ``fc`` layers then map ``400 -> 120 -> 84 -> 10``. The final
        10 outputs are raw CIFAR-10 class scores, also called logits.

    ReLU activations:
        ``F.relu`` applies ``relu(x) = max(0, x)`` after ``conv1``, ``conv2``, ``fc1``, and ``fc2``.
        Negative responses become 0, while positive responses stay positive. This makes each feature
        map easier to interpret as "this learned pattern fired here" versus "it did not fire much."
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class scores for a batch shaped ``(batch, 3, 32, 32)``.

        The forward pass mirrors the shape story in the class docstring: two convolution/ReLU/pool
        blocks produce inspectable feature maps, then the flattened values flow through the fully
        connected classifier head.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
