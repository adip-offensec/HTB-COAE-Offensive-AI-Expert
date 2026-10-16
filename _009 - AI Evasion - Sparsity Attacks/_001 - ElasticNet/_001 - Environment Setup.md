
ElasticNet requires significantly more computation than single-step attacks like FGSM. Each FISTA iteration performs forward and backward passes, with hundreds of iterations nested inside 5-10 binary search steps.

The `HTB Evasion Library` provides standardized infrastructure across all attack implementations: model architectures, MNIST data loaders, training loops, and visualization utilities. This consistency lets us focus on attack mechanics rather than boilerplate.

Code: bash

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

The setup follows a familiar pattern. We use `set_reproducibility(1337)` to ensure deterministic behavior across runs by fixing random seeds for PyTorch, NumPy, and CUDA. Device configuration automatically detects GPU availability, falling back to CPU when necessary. The HTB color constants (`HTB_GREEN`, `NODE_BLACK`, etc.) maintain visual consistency across all visualizations, with `use_htb_style()` applying the theme globally to matplotlib.

## Quick Setup Reference

Code: python

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities from HTB Evasion Library
from htb_ai_library.utils import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
    VIVID_PURPLE,
    AQUAMARINE,
)
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.training import train_model, evaluate_accuracy
from htb_ai_library.visualization import use_htb_style

# Apply HTB theme globally to all plots
use_htb_style()

# Set reproducibility
set_reproducibility(1337)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Example output (device names will vary):

Code: txt

```txt
Using device: cuda
GPU: NVIDIA GeForce RTX 5090
```

## Building the Target Model

ElasticNet targets `MNISTClassifierWithDropout`, a convolutional architecture with 2 conv layers (dropout rate 0.25) and 2 fully connected layers (dropout rate 0.5). Dropout serves dual purposes: preventing overfitting during training and creating a more realistic target that reflects production models with regularization.

Training from scratch takes several minutes even on GPU. When experimenting with attack parameters, retraining repeatedly wastes resources. The checkpoint caching approach checks for an existing `mnist_target.pth` file. If found, we load the saved weights directly. If absent, we train for 5 epochs and save the result. This makes the first run slower (training required) but all subsequent runs instant (weights loaded from disk).

Code: python

```python
# Get data loaders using library function
train_loader, test_loader = get_mnist_loaders(batch_size=128)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Create output directory for saving models and results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Define model checkpoint path in output directory
model_path = output_dir / "mnist_target.pth"

# Initialize model using MNISTClassifierWithDropout from library
model = MNISTClassifierWithDropout(num_classes=10).to(device)

# Check if trained model exists, otherwise train from scratch
if model_path.exists():
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"\nNo existing model found. Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, device=device)
    print(f"Saving trained model to {model_path}")
    save_model(model, model_path)

# Evaluate the trained model
accuracy = evaluate_accuracy(model, test_loader, device)
print(f"\nTest accuracy: {accuracy:.2f}%")
```

Expected output (first run, training from scratch):

Code: txt

```txt
Training samples: 60000
Test samples: 10000

No existing model found. Training new model...
Epoch 1/5: Avg Loss = 0.2761, Test Accuracy = 98.07%
Epoch 2/5: Avg Loss = 0.0982, Test Accuracy = 98.64%
Epoch 3/5: Avg Loss = 0.0750, Test Accuracy = 98.75%
Epoch 4/5: Avg Loss = 0.0616, Test Accuracy = 98.90%
Epoch 5/5: Avg Loss = 0.0529, Test Accuracy = 98.89%
Saving trained model to output/mnist_target.pth

Test accuracy: 98.89%
```

Expected output (subsequent runs, loading existing model):

Code: txt

```txt
Training samples: 60000
Test samples: 10000

Loading existing model from output/mnist_target.pth

Test accuracy: 98.89%
```

### Memory and Storage Details

How does batching affect memory usage? Each MNIST image occupies 28×28 = 784 pixels × 4 bytes (float32) = 3,136 bytes (~3.06 KiB). A batch of 128 images requires 128 × 3,136 bytes = 401,408 bytes (~392 KiB) for input data alone. Adding gradients, activations, and optimizer state for a CNN with 2 conv layers and 2 FC layers brings total per-batch memory to roughly 50-80 MB. Doubling the batch size to 256 would double memory usage to 100-160 MB while providing more stable gradient estimates but slower per-epoch iteration. The 128-image batch size balances these tradeoffs effectively for MNIST.

What gets serialized in the checkpoint file? When `save_model` writes `mnist_target.pth`, it creates a Python pickle containing an OrderedDict that maps parameter names (like `conv1.weight`, `fc2.bias`) to their tensor values. This `.pth` file stores all convolutional filters, batch norm statistics, and fully connected weights as floating-point arrays. When `load_model` reads this file, PyTorch's deserialization restores the exact numerical values from the previous training session, down to the last floating-point bit. This ensures perfect reproducibility across sessions while eliminating redundant training.

The final test accuracy of 98.89% (9,889 out of 10,000 examples classified correctly) confirms the model learned effectively while remaining well-regularized. With 98.89% accuracy, we have 9,889 potential attack targets in the test set, providing ample examples for evaluating ElasticNet's effectiveness against properly regularized neural networks.

