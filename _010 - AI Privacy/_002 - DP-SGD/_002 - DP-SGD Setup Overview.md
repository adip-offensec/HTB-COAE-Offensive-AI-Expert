
# Setup Overview

With the theoretical foundation established, we now move to implementation. Before training DP-SGD models, we need to configure our environment and understand the library components that handle privacy accounting behind the scenes. We use the `htb_ai_library` package for model training, evaluation, and visualization, which lets us focus on the privacy mechanisms rather than boilerplate code.

## Installing Dependencies

Install the HTB AI library from GitHub:

Code: bash

```bash
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

## Imports

Start your training script with these imports:

Code: python

```python
import os
import json
import torch
import torch.optim as optim
from safetensors.torch import save_file

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    CIFAR10CNN,
    get_cifar10_loaders,
    train_baseline_sgd,
    train_dp_sgd,
    evaluate_accuracy,
    compute_mia_advantage,
    plot_accuracy_comparison,
    plot_privacy_utility_tradeoff,
)
```

#### Library Components

The `htb_ai_library` handles core functionality for training and evaluating DP-SGD models. Here's what each component provides.

Reproducibility matters for meaningful comparisons. `set_reproducibility(seed)` ensures comparable results when measuring the privacy-utility tradeoff across different epsilon values.

Our classifier is `CIFAR10CNN`, a convolutional neural network designed for `CIFAR-10` image classification. It uses standard `Conv2d`, `ReLU`, `MaxPool2d`, and `Linear` layers, all compatible with Opacus `per-sample gradient` computation. `BatchNorm` layers are intentionally excluded because they would break DP-SGD's privacy guarantees (see the architecture constraints discussion in the previous section).

To load our data, we call `get_cifar10_loaders(batch_size, download)`, which fetches the `CIFAR-10` dataset (60,000 32x32 color images across 10 classes), applies standard normalization transforms, and returns a tuple of `(train_dataset, test_dataset, train_loader, test_loader)`. Each DP model needs its own loader instance because Opacus wraps loaders to track sample access for privacy accounting. Sharing loaders between models would corrupt privacy budget calculations, so we call `get_cifar10_loaders()` separately for each epsilon configuration.

Training happens through two functions. `train_baseline_sgd(model, train_loader, device, epochs, learning_rate, momentum)` runs standard SGD training over fixed epochs, returning the trained model. `train_dp_sgd(model, train_loader, privacy_engine, optimizer, device, epochs, delta)` trains with DP-SGD using an attached Opacus `privacy engine`. The `privacy_engine` and `optimizer` must both reference the same model since Opacus coordinates them internally to ensure gradient clipping and noise addition happen correctly. This function returns `(model, final_epsilon)`.

To evaluate our models, we use two metrics. We call `evaluate_accuracy(model, loader, device)` to compute classification accuracy on a dataset (returning a percentage from 0-100). To measure membership inference vulnerability, we use `compute_mia_advantage(model, train_loader, test_loader, device)`, which performs a confidence-threshold attack and returns `(attack_accuracy, advantage)` where advantage is accuracy minus 0.5.

To visualize our results, we use two functions. We call `plot_accuracy_comparison()` to generate a grouped bar chart comparing test accuracy across baseline, ε=10, and ε=3 models. For a comprehensive view of the tradeoff, we use `plot_privacy_utility_tradeoff()` to create a dual-axis plot showing accuracy and `MIA` advantage across privacy levels.

#### Configuration Parameters

Code: python

```python
RANDOM_SEED = 1337
BATCH_SIZE = 256
BASELINE_EPOCHS = 20
BASELINE_LR = 0.1
DP_EPOCHS = 20
DP_LR = 0.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

set_reproducibility(RANDOM_SEED)
use_htb_style()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("figs", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
```

We keep training conditions parallel between baseline and DP-SGD (same epochs, batch size, learning rate) so that differences in final accuracy reflect the privacy mechanism itself rather than training configuration. `RANDOM_SEED = 1337` ensures reproducibility, enabling meaningful comparisons between runs.

The DP-specific parameters are `MAX_GRAD_NORM = 1.0`, which bounds each sample’s gradient contribution before noise addition, and `DELTA = 1e-5`, which represents the privacy failure probability in our (ε,δ)-differential privacy guarantee. For guidance on choosing these values, see the hyperparameter tuning discussion in the Overview.

Three directories organize our outputs: `OUTPUT_DIR` stores intermediate checkpoints and results, `FIGS_DIR` holds visualization figures, and `MODELS_DIR` stores the final safetensors model for validation server submission.

## Simplified "MIA" Measurement

The previous section used the full shadow model attack methodology. Here we use a simplified confidence-threshold approach as a measurement tool. Using `compute_mia_advantage()`, we collect maximum softmax confidence for all training samples (members) and all test samples (non-members), then balance these samples (10,000 from each) to avoid class imbalance. Why 10,000? `CIFAR-10` has exactly 10,000 test samples, so we match that count from the 50,000 training samples for fair comparison. With unbalanced datasets, this approach would need adjustment (either subsampling the larger set or using stratified sampling).

It searches for the optimal threshold that maximizes attack accuracy, returning `(attack_accuracy, advantage)` where advantage is accuracy minus 0.5. This captures the same underlying signal (overfitted models are more confident on training data) without the overhead of training shadow models.

