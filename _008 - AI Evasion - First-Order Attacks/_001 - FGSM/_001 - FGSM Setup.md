
# FGSM Setup

## Library Installation

This module uses the HTB Evasion Library which provides common utilities. Install it first:

Code: bash

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

## Environment Setup

Reproducibility matters. Without it, gradient computations can vary between runs even on identical inputs, making debugging impossible and comparisons meaningless. The Library provides shared utilities to eliminate these sources of randomness while keeping the setup minimal:

Code: python

```python
import os
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import common utilities from HTB Evasion Library
from htb_ai_library import (
    set_reproducibility,
    SimpleCNN,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy
)

# Configure reproducibility
set_reproducibility(1337)
```

How does `set_reproducibility` enforce determinism across multiple libraries? It controls three independent randomness sources. First, the `PYTHONHASHSEED` environment variable locks Python's built-in hash randomization, ensuring dictionary iteration order remains stable. Second, both Python's `random` module and NumPy's generator receive the seed `1337`, synchronizing their outputs. Third, PyTorch's cuDNN backend (which normally chooses algorithms based on runtime heuristics) switches to deterministic mode, trading a small speed penalty for perfect reproducibility across CPU and GPU.

## Data and Model Architecture

Why MNIST? Speed and clarity. Training a ResNet-50 on ImageNet takes hours and produces results hard to visualize. MNIST digits fit in a terminal, train in seconds, and still reveal vulnerabilities. We'll use a compact convolutional classifier that reaches 98% accuracy in one epoch, establishing a strong baseline while FGSM materially reduces robustness.

### Device Configuration

PyTorch operations need a target device (CPU or GPU). Checking availability first ensures code runs everywhere, from laptops to cloud instances with CUDA:

Code: python

```python
# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The conditional `torch.cuda.is_available()` returns `True` when NVIDIA drivers and CUDA toolkit are properly installed. GPU acceleration typically speeds training by 10-50x for convolutional networks, but our small MNIST model trains fast enough on CPU that device choice won't significantly affect the demonstration.

### Data Loading

The library's data loader handles MNIST dataset preparation, conversion, and batching:

Code: python

```python
# Prepare data loaders using library function (normalized space)
train_loader, test_loader = get_mnist_loaders(batch_size=128, normalize=True)
```

What does `get_mnist_loaders` do? It applies `transforms.ToTensor()` to convert each PIL image into a PyTorch tensor while rescaling from [0, 255] to [0, 1]. When `normalize=True` is specified, it applies MNIST-specific normalization that transforms pixels to a normalized space. The function downloads MNIST to `./data` if not already present. Crucially, it instantiates the training data loader's shuffling generator with seed 1337, guaranteeing that batch 0 always contains the same 128 samples. The function sets `num_workers=0` to avoid multiprocessing race conditions and disables `pin_memory` for simplicity.

### Model Definition

The architecture needs enough capacity to learn MNIST reliably but not so much that training becomes slow. Two convolutional layers extract spatial features, followed by two fully connected layers for classification:

Code: python

```python
# Initialize model using library's SimpleCNN
model = SimpleCNN().to(device)
```

What happens inside `SimpleCNN`? The first convolutional layer (`conv1`) processes the single-channel 28×28 input with 32 filters of size 3×3. Setting `padding=1` maintains spatial dimensions at 28×28 after convolution. ReLU activation introduces nonlinearity. The second convolutional layer (`conv2`) applies 64 filters with the same kernel size, doubling the channel count to capture richer feature representations.

Two max-pooling operations (each with stride 2) reduce spatial resolution from 28×28 to 14×14, then from 14×14 to 7×7. This gives a final feature map of shape 64×7×7. The `view` operation reshapes this into a flat 3136-dimensional vector (since 64×7×7=3136). Two fully connected layers process this vector, with the final layer outputting 10 raw logits corresponding to the digit classes 0-9.

The `.to(device)` call moves all model parameters to the configured device (GPU if available, CPU otherwise). This ensures consistency between where the model lives and where input data will be sent during training.

### Training and Evaluation

Training and evaluation follow standard supervised learning patterns. The library encapsulates these in `train_model` and `evaluate_accuracy` to keep code focused on attacks rather than infrastructure:

Code: python

```python
# Train the model using library function
trained_model = train_model(model, train_loader, test_loader, epochs=1, device=device)

# Evaluate baseline accuracy using library function
baseline_acc = evaluate_accuracy(trained_model, test_loader, device)
print(f"Baseline test accuracy: {baseline_acc:.2f}%")
```

The `train_model` function orchestrates the complete training loop. Each epoch begins with `model.train()`, enabling dropout (if present) and switching batch normalization to training mode where it updates running statistics. The optimizer (Adam with default learning rate 0.001) receives `set_to_none=True` during gradient clearing, which deallocates gradient tensors entirely rather than zeroing them in place, saving memory and time. Loss accumulates across batches weighted by batch size, producing an accurate epoch-level average. After all epochs finish, the function returns the updated model.

How does `evaluate_accuracy` differ from training? It switches the model to evaluation mode via `model.eval()`, which freezes batch normalization statistics and disables dropout. The `torch.no_grad()` context manager prevents gradient computation, reducing memory usage and accelerating inference. For each batch, `argmax(dim=1)` extracts the highest-scoring class index from the 10-dimensional logit vector. Accuracy is the fraction of correct predictions across the entire test set.

Does one epoch suffice? For MNIST, yes. The dataset is small (60,000 training images) and the patterns are simple (isolated digits on blank backgrounds). A single pass through the data typically converges to 98-99% accuracy. Training longer yields diminishing returns, maybe reaching 99.2%, but the added accuracy doesn't change the main point: even a well-trained model remains vulnerable to adversarial perturbations.

Expected output:

Code: txt

```txt
Epoch 1/1: Avg Loss = 0.1566, Test Accuracy = 98.41%
Baseline test accuracy: 98.41%
```

The high accuracy (98.41%) on clean test data establishes a strong baseline for us to work with.

