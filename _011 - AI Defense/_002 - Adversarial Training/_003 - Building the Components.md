
# Building the Components

This section implements the core components needed for adversarial training: the model architecture, attack functions, and evaluation utilities. You will build a LeNet-5 classifier, implement FGSM and I-FGSM attacks, create evaluation functions for measuring robustness, and train a baseline model that demonstrates vulnerability to adversarial examples.

---

## Project Setup

To build our adversarial training system, we start with the required imports. We use `htb_ai_library` for common utilities that handle reproducibility, data loading, and evaluation:

  Building the Components

```shell-session
# Install the AI Library (or update it)
@htb[/htb]$ pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

Afterward, we can prepare the required imports:

Code: python

```python
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import save_file
from tqdm import tqdm

from htb_ai_library import (
    set_reproducibility,
    get_mnist_loaders,
    evaluate_accuracy,
    train_model,
)
```

We get `set_reproducibility` for deterministic training, `get_mnist_loaders` for MNIST data loading, `evaluate_accuracy` for model evaluation, and `train_model` for standard baseline training. We use `safetensors` for secure, efficient model serialization.

To configure our training environment, we define constants for MNIST normalization and epsilon bounds:

Code: python

```python
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
EPSILON = 0.3
EPSILON_SPREAD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
I_FGSM_STEPS = 10


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

We use `MNIST_MEAN` and `MNIST_STD` to compute FGSM clamping bounds, ensuring perturbed pixels stay within the valid normalized range `[-0.4242, 2.8215]`. `EPSILON` defines the primary attack budget (0.3) for evaluation, while `EPSILON_SPREAD` lists perturbation magnitudes from 0.1 to 1.0 for comprehensive robustness training. `I_FGSM_STEPS` controls how many iterations we use in the iterative FGSM attack during evaluation.

---

## Saving Adversarial Examples

Safetensors provides secure, efficient model serialization but only supports flat dictionaries of tensors. To save adversarial examples with their metadata (epsilon values, labels), we need a helper function that converts nested dictionaries to the flat structure safetensors expects:

Code: python

```python
def save_adversarial_examples(data, path):
    """Save adversarial examples to safetensors format."""
    # We don't store fgsm_images/ifgsm_images separately since they reference
    # fgsm_by_epsilon[epsilon] and would cause memory sharing errors in safetensors
    tensors = {
        'clean_images': data['clean_images'],
        'clean_labels': data['clean_labels'].long(),
    }

    for eps in data['epsilon_spread']:
        key = f"{eps:.1f}"
        tensors[f'fgsm_eps_{key}'] = data['fgsm_by_epsilon'][eps]
        tensors[f'ifgsm_eps_{key}'] = data['ifgsm_by_epsilon'][eps]

    metadata = {
        'epsilon': str(data['epsilon']),
        'epsilon_spread': json.dumps(data['epsilon_spread']),
    }

    save_file(tensors, path, metadata=metadata)
```

We flatten nested epsilon dictionaries into keys like `fgsm_eps_0.3` and `ifgsm_eps_0.3`, storing scalar metadata (epsilon value, spread list) as JSON strings in the safetensors metadata field. We avoid storing `fgsm_images` and `ifgsm_images` separately since they reference the same tensors as their epsilon dictionary entries, which would cause safetensors memory sharing errors.

---

## Model Architecture

To classify MNIST digits, we use a LeNet-5 architecture that processes 28x28 grayscale images through two convolutional layers with max pooling, followed by three fully connected layers producing 10-class logits:

Code: python

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

We must keep this architecture unchanged because the evaluator expects exactly this structure. Input images flow through `conv1`, producing 6 feature maps with shape 28x28 (padding preserves dimensions). ReLU activation and 2x2 max pooling then reduce spatial dimensions to 14x14. A second convolution-pool block produces 16 feature maps at 5x5, which we flatten to 400 values (16 * 5 * 5) and pass through three fully connected layers. The final layer outputs raw logits for the 10 digit classes, with no softmax (we apply that during loss computation).

We also need to understand the FGSM attack function, which computes adversarial perturbations using the gradient of the loss with respect to input pixels:

Code: python

```python
def fgsm_attack(model, images, labels, epsilon):
    images_copy = images.clone().detach().requires_grad_(True)
    outputs = model(images_copy)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images_copy.grad.sign()
    adv_images = images_copy + epsilon * grad_sign
    min_val = (0 - MNIST_MEAN) / MNIST_STD
    max_val = (1 - MNIST_MEAN) / MNIST_STD
    adv_images = torch.clamp(adv_images, min_val, max_val)
    return adv_images.detach()
```

Why clone and detach before enabling gradients? Cloning prevents corruption of the original batch, while `detach()` breaks the computational graph so gradients do not flow backward through the attack during training. Notice `model.zero_grad()` before the backward pass: without this, gradients accumulate from previous iterations, producing incorrect perturbation directions. The clamping bounds `[-0.4242, 2.8215]` come from MNIST normalization (mean 0.1307, std 0.3081), ensuring perturbed pixels stay within the valid normalized range. Forgetting to clamp can produce out-of-distribution inputs that behave unpredictably.

The evaluator also tests against I-FGSM (Iterative FGSM), which applies multiple smaller FGSM steps to find stronger adversarial examples. To generate these evaluation examples, we iteratively refine the perturbation direction:

Code: python

```python
def i_fgsm_attack(model, images, labels, epsilon, steps=I_FGSM_STEPS):
    """Generate I-FGSM (Iterative FGSM) adversarial examples."""
    alpha = epsilon / steps  # Step size per iteration
    min_val = (0 - MNIST_MEAN) / MNIST_STD
    max_val = (1 - MNIST_MEAN) / MNIST_STD

    adv_images = images.clone().detach()
    original_images = images.clone().detach()
```

We set step size `alpha` to `epsilon / steps`, so 10 steps of size 0.03 each achieve the full epsilon 0.3 budget. Storing `original_images` separately lets us project perturbations back to the epsilon-ball after each iteration.

Code: python

```python
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign
```

Each iteration recomputes gradients at the current adversarial position, finding the locally worst direction. We take a small step `alpha` in that direction, then detach to prevent gradient accumulation across iterations.

Code: python

```python
        # Project back to epsilon-ball around original
        perturbation = adv_images - original_images
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_images = original_images + perturbation

        # Clamp to valid range
        adv_images = torch.clamp(adv_images, min_val, max_val)

    return adv_images.detach()
```

Projection is essential: without it, cumulative perturbations could exceed the epsilon budget. We compute how far we have strayed from the original, clamp that difference to `[-epsilon, epsilon]`, then reconstruct `adv_images` within bounds. A final clamp ensures pixel values stay in the valid normalized range.

---

## Evaluation Functions

We imported `evaluate_accuracy` from `htb_ai_library`, which handles clean accuracy evaluation. To measure adversarial robustness, we need a custom function that generates fresh FGSM attacks against the current model and counts correctly classified perturbed inputs:

Code: python

```python
def evaluate_adversarial_accuracy(model, loader, device, epsilon, num_batches=None):
    """Evaluate accuracy under FGSM attack."""
    model.eval()
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)
```

We initialize counters and iterate through the data loader. The optional `num_batches` parameter lets us evaluate a subset for speed during training. Setting `num_batches=20` tests approximately 2,500 samples, enough for reliable accuracy estimates without processing the entire test set.

Code: python

```python
        # Generate adversarial examples (need gradients, so briefly enable train mode)
        model.train()
        adv_images = fgsm_attack(model, images, labels, epsilon)
        model.eval()

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total
```

Why switch to `model.train()` for attack generation? FGSM requires gradient computation, which needs training mode for batch normalization layers (though LeNet-5 lacks these, the pattern ensures correctness for any architecture). We switch back to `model.eval()` for the actual prediction, then accumulate correct counts and return the percentage.

---

## Baseline Training

Before adversarial training, we need a baseline model trained with standard (non-adversarial) methods. This baseline serves two purposes: it provides a comparison point to measure defense improvement, and it generates adversarial examples for consistent evaluation across training runs.

We use `train_model` from `htb_ai_library`, which handles the standard training loop with Adam optimizer at learning rate 0.001. The function trains for a specified number of epochs and prints progress after each epoch:

Code: python

```python
train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir="./data")
baseline_model = LeNet5()
baseline_model = train_model(
    baseline_model,
    train_loader,
    test_loader,
    device=get_device(),
    epochs=10,
    learning_rate=0.001,
)
```

As established in the introduction, standard training achieves high clean accuracy but provides no adversarial robustness, making the model our starting point for adversarial training.

With all components in place (model architecture, attack functions, evaluation utilities, and baseline model), the next section generates adversarial examples for consistent evaluation and implements the adversarial training loop that transforms this vulnerable baseline into a robust classifier.

