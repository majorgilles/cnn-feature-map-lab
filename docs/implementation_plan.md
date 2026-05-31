# Implementation Plan

## Goal

Build a 5-day, notebook-led learning repo that makes CNN feature maps concrete through visual experiments, a tiny PyTorch CNN, and a brief pretrained-model comparison.

## Milestones

### Day 1 — Manual filters make feature maps concrete

- Load one or two sample RGB images.
- Apply simple edge, blur, and sharpen kernels.
- Save visual grids under `outputs/day01_manual_filters/`.
- Write a short note: what did each filter highlight?

### Day 2 — PyTorch `Conv2d` and `MaxPool2d`

- Recreate simple filters with `torch.nn.Conv2d`.
- Show input channels, output channels, kernels, and feature-map shapes.
- Apply `MaxPool2d` and compare before/after feature maps.
- Save outputs under `outputs/day02_conv2d_pooling/`.

### Day 3 — Tiny CIFAR-10 CNN

- Follow the official PyTorch CIFAR-10 classifier tutorial as the backbone.
- Keep the model and training run small.
- Treat accuracy as a sanity check, not the learning goal.
- Save a checkpoint under `models/`.

### Day 4 — Inspect tiny CNN activations

- Register hooks or return intermediate activations from the tiny CNN.
- Run a few real images or CIFAR-10 examples through the model.
- Save activation grids under `outputs/day04_tiny_cnn_activations/`.
- Write observations about early vs deeper layers.

### Day 5 — Pretrained CNN comparison

- Use one torchvision pretrained model.
- Extract a few intermediate activations using torchvision feature extraction utilities.
- Compare the real model’s activations with the tiny CNN’s activations.
- Write the final 5-10 sentence explanation of feature maps.

## Validation

Done means the repo contains an output gallery with manual filters, `Conv2d` feature maps, pooling comparisons, tiny CNN activations, pretrained CNN activations, and a short learner-authored explanation.
