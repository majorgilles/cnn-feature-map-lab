# Tutorial Outline

## 1. Manual filters

Begin with the simplest possible mental model: a kernel scans across an image and produces a new grid. That grid is the beginner-friendly version of a feature map.

## 2. `Conv2d` and pooling

Connect the manual filter idea to PyTorch. `Conv2d` produces feature maps; `MaxPool2d` transforms feature maps by shrinking them and keeping strong responses.

## 3. Tiny CNN on CIFAR-10

Use the official PyTorch CIFAR-10 tutorial as the model/data backbone. The classifier gives the CNN a reason to learn useful filters, but accuracy is only a sanity check.

## 4. Activation inspection

Look inside the tiny CNN. Compare feature maps from early and later layers, and write short observations about what changes with depth.

## 5. Pretrained CNN comparison

Use torchvision feature extraction to inspect a few activations from a pretrained CNN. The goal is to connect the toy model to real image CNNs.
