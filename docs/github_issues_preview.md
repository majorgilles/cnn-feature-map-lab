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

## Issue 2.5 — Bridge: Train a tiny synthetic-orientation CNN before CIFAR-10

Labels: `learning`, `pytorch`, `feature-maps`, `bridge`, `day-2.5`

### Goal

Build a small bridge notebook between fixed manual kernels and the full CIFAR-10 classifier. Use simple synthetic grayscale images and a tiny CNN to show how learned convolution filters become a small feature hierarchy before introducing RGB images, 10 classes, and the larger CIFAR training workflow.

### Official / authoritative anchors

- PyTorch `torch.nn.Conv2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch `torch.nn.MaxPool2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- PyTorch `torch.nn.CrossEntropyLoss` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Supplemental blueprint for simple orientation/grating stimuli: Neuromatch Tutorial 2, Convolutional Neural Networks: https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial2.html
- Supplemental blueprint for a simple orientation-discrimination CNN: Neuromatch Tutorial 3, orientation discrimination task: https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial3.html

### Why it matters

Day 2 explains one fixed kernel and one pooled feature map. Day 3 jumps to real CIFAR-10 images, RGB channels, multiple convolution layers, fully connected layers, loss, optimizer, training loop, and checkpointing. This bridge should isolate the next mental step: a CNN can learn several simple filters from a tiny visual task, and a second convolution layer can combine first-layer feature maps into slightly richer patterns.

### Proposed notebook

Create `notebooks/02b_tiny_orientation_cnn.ipynb`.

The notebook should sit conceptually between:

```text
Day 2: fixed kernel -> one feature map -> pooling
Day 2.5: synthetic grayscale images -> tiny learned CNN -> inspect learned filters
Day 3: CIFAR-10 RGB batches -> tutorial CNN -> checkpoint for later feature-map inspection
```

### Steps

- Create a project-relative output directory: `outputs/day02b_tiny_orientation_cnn/`.
- Record the exact Windows PowerShell command used to open or run the notebook, for example:

  ```powershell
  uv run jupyter lab notebooks/02b_tiny_orientation_cnn.ipynb
  ```

- Generate a tiny synthetic grayscale dataset inside the notebook, with no external downloads.
  - Use `32 x 32` single-channel images.
  - Use two visually simple classes, such as left-tilted vs right-tilted bars/gratings, or vertical vs horizontal stripe patterns.
  - Add only mild variation, such as small shifts, line thickness changes, or light noise, so the task remains inspectable.
  - Keep the generation code small, seeded, and readable.
- Show a labeled sample grid before training.
- Wrap the generated tensors in a simple `TensorDataset` and `DataLoader` so the learner sees the same data-loader pattern used later without CIFAR-10 complexity.
- Define a deliberately tiny CNN with explicit keyword arguments, for example:

  ```python
  class TinyOrientationCNN(nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
          self.classifier = nn.Linear(in_features=8 * 8 * 8, out_features=2)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = torch.flatten(x, 1)
          x = self.classifier(x)
          return x
  ```

- Keep the architecture intentionally smaller than the CIFAR tutorial:
  - one input channel instead of three RGB channels
  - two classes instead of ten
  - two small convolution layers, but only one final linear classifier
  - no extra `fc1 -> fc2 -> fc3` classifier head
  - no checkpoint requirement unless it feels useful later
- Add a dedicated section titled `## Trace shapes through the tiny CNN`, modeled after the step-by-step shape walk-through in `notebooks/03_tiny_cifar10_cnn.ipynb`.
  - Use separate markdown + code cells for each explicit transformation, not one long combined cell.
  - Include at least: input batch, `conv1`, `relu1`, `pool1`, `conv2`, `relu2`, `pool2`, `flatten`, and `classifier`.
  - Each markdown cell should say what changes at that step: channels, height/width, values-only ReLU, or vector length.
  - Each code cell should update `x` once and print `x.shape`.
  - Expected path:

    ```text
    input:      [batch, 1, 32, 32]
    conv1:      [batch, 4, 32, 32]
    relu1:      [batch, 4, 32, 32]
    pool1:      [batch, 4, 16, 16]
    conv2:      [batch, 8, 16, 16]
    relu2:      [batch, 8, 16, 16]
    pool2:      [batch, 8, 8, 8]
    flatten:    [batch, 512]
    classifier: [batch, 2]
    ```

- Train for a short run using `CrossEntropyLoss` and a simple optimizer such as SGD or Adam.
- Treat accuracy as a sanity check only; the real learning goal is whether the filters and activations become interpretable.
- Save at least these artifacts under `outputs/day02b_tiny_orientation_cnn/`:
  - a synthetic sample grid
  - a training loss curve or small loss table
  - a learned `conv1` filter grid
  - at least one activation grid from `conv1` or `conv2`
- Add a reflection section answering:
  - Which part felt like Day 2 fixed kernels?
  - Which part starts to feel like Day 3 CNN training?
  - What did the second convolution layer add conceptually?

### Human-in-the-loop checkpoint

- [ ] I inspected the synthetic examples before training.
- [ ] I stepped through the dedicated shape-tracing section cell by cell.
- [ ] I confirmed that training ran and used accuracy only as a sanity check.
- [ ] I inspected the learned filters or activation maps myself.
- [ ] I wrote why this bridge makes the CIFAR-10 notebook feel less abrupt.

### Acceptance criteria

- [ ] `notebooks/02b_tiny_orientation_cnn.ipynb` exists and runs from a fresh `uv sync` environment.
- [ ] The notebook cites the PyTorch docs for `Conv2d`, `MaxPool2d`, and `CrossEntropyLoss`.
- [ ] The notebook clearly labels the Neuromatch orientation/grating material as supplemental inspiration, not a replacement for this repo's learning path.
- [ ] The synthetic dataset is generated locally in the notebook with no external dataset download.
- [ ] The CNN uses a tiny grayscale, two-class architecture with no more than two convolution layers and one final linear classifier.
- [ ] The notebook has a dedicated `## Trace shapes through the tiny CNN` section with separate markdown/code cells for each shape transformation.
- [ ] Tensor shapes are printed and explained through the forward path.
- [ ] A short training run completes successfully.
- [ ] At least one learned-filter grid or activation grid is saved under `outputs/day02b_tiny_orientation_cnn/`.
- [ ] The notebook explicitly explains that this is a bridge from fixed kernels to learned feature maps, not a classifier-focused detour.

### Non-goals

- Do not introduce CIFAR-10 in this bridge notebook.
- Do not add a pretrained model.
- Do not use a large classifier head or more than two convolution layers.
- Do not optimize for high benchmark accuracy.

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

Collect the learning evidence and create a final explanation that is AI-summarized from the learner's inspection and discussion.

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
- [ ] I reviewed the AI-summarized feature-map explanation against the gallery and my learning notes.
- [ ] I decided whether to proceed, rerun, ask for help, or open a follow-up issue.

### Acceptance criteria

- [ ] Manual filter grids are represented in the gallery.
- [ ] Conv2d feature maps are represented in the gallery.
- [ ] Pooling before/after is represented in the gallery.
- [ ] Tiny CNN activations are represented in the gallery.
- [ ] Pretrained CNN activations are represented in the gallery.
- [ ] A 5-10 sentence final explanation is present.
