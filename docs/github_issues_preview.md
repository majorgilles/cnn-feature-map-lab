# GitHub Issues Preview

These issues are ready to create after review. Proposed labels are listed for each issue; labels should be created if they do not already exist.

## Issue 1 — Day 1: Make feature maps concrete with manual image filters

Labels: `learning`, `pytorch`, `feature-maps`, `day-1`

### Goal

Build the first visual intuition for feature maps by applying simple image filters to one or two real images.

### Official / authoritative anchors

- PyTorch `torch.nn.Conv2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

### Why it matters

Before using a trained CNN, make the core idea visible: a small kernel scans across an image and produces an output grid. That output grid is the beginner-friendly version of a feature map.

### Steps

- Open `notebooks/01_manual_filters.ipynb`.
- Add or download one or two small sample RGB images under `data/` and document their source.
- Implement or use simple fixed kernels: edge, blur, sharpen.
- Show the original image beside each filtered output.
- Save a visual grid under `outputs/day01_manual_filters/`.
- Tweak one kernel value and rerun the relevant cell.

### Human-in-the-loop checkpoint

- [ ] I inspected the filter outputs myself.
- [ ] I can point to a filtered output and explain what pattern became stronger.
- [ ] I wrote a short reflection in the notebook.

### Acceptance criteria

- [ ] Notebook runs from a fresh `uv sync` environment.
- [ ] At least one output grid is saved under `outputs/day01_manual_filters/`.
- [ ] The notebook explains why the output grid can be thought of as a feature map.
- [ ] One kernel tweak and observation are recorded.

---

## Issue 2 — Day 2: Connect manual filters to PyTorch Conv2d and MaxPool2d

Labels: `learning`, `pytorch`, `feature-maps`, `day-2`

### Goal

Use PyTorch layers to reproduce the manual-filter idea and clearly distinguish convolution-produced feature maps from pooling-transformed feature maps.

### Official / authoritative anchors

- PyTorch `torch.nn.Conv2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch `torch.nn.MaxPool2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

### Why it matters

`Conv2d` produces feature maps by applying learned or fixed filters. `MaxPool2d` does not usually discover a new pattern; it shrinks feature maps and keeps strong local responses.

### Steps

- Open `notebooks/02_conv2d_pooling.ipynb`.
- Create a `Conv2d` layer with a fixed edge-detection kernel.
- Run one image through the layer and inspect output shape.
- Apply `MaxPool2d` to the feature map.
- Save before/after pooling visualizations under `outputs/day02_conv2d_pooling/`.
- Document the meaning of `[batch, channels, height, width]` for each tensor.

### Human-in-the-loop checkpoint

- [ ] I inspected convolution outputs and pooling outputs.
- [ ] I can explain the difference between “Conv2d produced this feature map” and “MaxPool2d resized/filtered this existing feature map.”
- [ ] I wrote a short reflection in the notebook.

### Acceptance criteria

- [ ] The notebook cites the official `Conv2d` and `MaxPool2d` docs.
- [ ] At least one Conv2d feature-map grid is saved.
- [ ] At least one pooling before/after comparison is saved.
- [ ] Tensor shapes are printed and explained.

---

## Issue 3 — Day 3: Train a tiny CIFAR-10 CNN as a feature-map learning vehicle

Labels: `learning`, `pytorch`, `cifar-10`, `day-3`

### Goal

Follow the official PyTorch CIFAR-10 classifier tutorial closely enough to train a tiny CNN, while keeping classification as a vehicle for learned filters rather than the main topic.

### Official / authoritative anchors

- PyTorch CIFAR-10 tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

### Why it matters

Manual filters are useful, but CNN filters become more meaningful after they learn from data. A tiny classifier gives the CNN a reason to learn visual patterns that can be inspected later.

### Steps

- Open `notebooks/03_tiny_cifar10_cnn.ipynb`.
- Follow the official CIFAR-10 tutorial structure for data loading, model definition, loss, optimizer, and training loop.
- Keep the model and number of epochs small.
- Use CUDA when available, with CPU fallback.
- Save a checkpoint under `models/`.
- Record exact Windows PowerShell commands used.

### Human-in-the-loop checkpoint

- [ ] I confirmed the training loop ran.
- [ ] I treated accuracy as a sanity check, not the objective.
- [ ] I wrote a short note about what the model has learned enough for us to inspect.

### Acceptance criteria

- [ ] The notebook cites and follows the official CIFAR-10 tutorial backbone.
- [ ] A tiny CNN trains successfully for a short run.
- [ ] A checkpoint is saved under `models/`.
- [ ] The notebook explains that classification is only being used to create learned feature maps.

---

## Issue 4 — Day 4: Inspect feature maps and activations from the tiny CNN

Labels: `learning`, `pytorch`, `feature-maps`, `day-4`

### Goal

Run images through the trained tiny CNN and save visualizations of intermediate activations.

### Official / authoritative anchors

- PyTorch `torch.nn.Module` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
- torchvision feature extraction docs for comparison/reference: https://docs.pytorch.org/vision/stable/feature_extraction.html

### Why it matters

This is where feature maps become less abstract: you can see how early layers respond to simple patterns and later layers respond to more model-specific structures.

### Steps

- Open `notebooks/04_feature_map_inspection.ipynb`.
- Load the tiny CNN checkpoint from `models/`.
- Capture activations from selected convolution layers.
- Visualize several channels from each activation tensor.
- Save activation grids under `outputs/day04_tiny_cnn_activations/`.
- Compare at least one early layer with one later layer.

### Human-in-the-loop checkpoint

- [ ] I inspected activation grids myself.
- [ ] I wrote what seems clearer in early layers versus later layers.
- [ ] I wrote one remaining question about feature maps.

### Acceptance criteria

- [ ] At least two layers have saved activation grids.
- [ ] The notebook explains activation tensor shape.
- [ ] The notebook explicitly uses the phrase “feature map” for individual activation channels or channel grids.
- [ ] A short observation note is included.

---

## Issue 5 — Day 5: Compare with a pretrained torchvision CNN

Labels: `learning`, `pytorch`, `torchvision`, `day-5`

### Goal

Use one pretrained torchvision model to inspect a few real CNN activations and compare them with the toy CNN.

### Official / authoritative anchors

- torchvision feature extraction docs: https://docs.pytorch.org/vision/stable/feature_extraction.html

### Why it matters

The tiny CNN teaches the mechanics, but a pretrained model shows that the same feature-map idea scales to real image networks.

### Steps

- Open `notebooks/05_pretrained_comparison.ipynb`.
- Load one pretrained torchvision CNN.
- Use torchvision feature extraction utilities to select a few internal nodes/layers.
- Run one or two sample images through the model.
- Save selected activation grids under `outputs/day05_pretrained_comparison/`.
- Compare the pretrained activations with the tiny CNN activations.

### Human-in-the-loop checkpoint

- [ ] I inspected pretrained-model activation grids myself.
- [ ] I wrote one similarity and one difference compared with the tiny CNN.
- [ ] I kept this as a brief comparison, not a new interpretability project.

### Acceptance criteria

- [ ] The notebook cites the official torchvision feature extraction docs.
- [ ] At least one pretrained activation grid is saved.
- [ ] The notebook includes a brief toy-vs-pretrained comparison note.

---

## Issue 6 — Final: Build the output gallery and write the feature-map explanation

Labels: `learning`, `documentation`, `feature-maps`, `final-note`

### Goal

Collect the learning evidence and write the final explanation in your own words.

### Official / authoritative anchors

- PyTorch CIFAR-10 tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- PyTorch `Conv2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch `MaxPool2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- torchvision feature extraction docs: https://docs.pytorch.org/vision/stable/feature_extraction.html

### Why it matters

The strongest proof of understanding is not a green notebook run. It is being able to inspect visual outputs and explain what the CNN layers are doing.

### Steps

- Gather the best images from `outputs/day01_*` through `outputs/day05_*`.
- Create a small gallery section in `README.md` or a final notebook note.
- Write a 5-10 sentence explanation of feature maps.
- Include one sentence distinguishing convolution outputs from pooling-transformed feature maps.
- Include exact Windows PowerShell commands used during setup/runs.

### Human-in-the-loop checkpoint

- [ ] I inspected the final gallery myself.
- [ ] I wrote the feature-map explanation in my own words.
- [ ] I decided whether to proceed, rerun, ask for help, or open a follow-up issue.

### Acceptance criteria

- [ ] Manual filter grids are represented in the gallery.
- [ ] Conv2d feature maps are represented in the gallery.
- [ ] Pooling before/after is represented in the gallery.
- [ ] Tiny CNN activations are represented in the gallery.
- [ ] Pretrained CNN activations are represented in the gallery.
- [ ] A 5-10 sentence final explanation is present.
