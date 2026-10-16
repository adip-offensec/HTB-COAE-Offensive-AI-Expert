
# Visualization

Visual analysis reveals how adversarial perturbations affect model predictions beyond numerical metrics. This section builds visualization infrastructure to display clean images, adversarial examples, perturbations, and probability shifts side by side. The visualizations help understand attack mechanisms and validate implementation correctness.

## Visualization Setup

To maintain consistent visual styling across all plots, we need a helper function that applies the theme to matplotlib axes:

Code: python

```python
import matplotlib.pyplot as plt
import numpy as np

# Colors imported from library
from htb_ai_library import (
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

def _style_axes(ax: plt.Axes) -> None:
    """Apply Hack The Box dark theme to an axes instance.

    Args:
        ax: Matplotlib axes to style
    """
    ax.set_facecolor(NODE_BLACK)
    ax.tick_params(colors=HACKER_GREY)
    for spine in ax.spines.values():
        spine.set_color(HACKER_GREY)
    ax.grid(True, color=HACKER_GREY, linestyle="--", alpha=0.25)
```

The function sets background color, tick colors, spine colors, and adds a subtle grid. The `alpha=0.25` creates a faint grid that doesn't dominate the visualization.

## Main Visualization Function

The visualization displays original, adversarial, and perturbation images alongside class probabilities. Let's build this step by step.

First, the function prepares the attack and computes predictions:

Code: python

```python
def visualize_attack(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    make_adv,
                    title: str,
                    num_classes: int = 10,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """HTB-styled visualization for adversarial examples.

    Args:
        model: Classifier in evaluation mode
        image: Single image in normalized space, shape (C,H,W)
        label: Scalar true label tensor
        make_adv: Callable (model, image_batch, label_batch) -> adv_batch in normalized space
        title: Figure title
        num_classes: Number of classes to show in probability bars
        targeted: Whether the attack is targeted
        target_class: Optional target class to annotate
    """
    model.eval()
    dev = next(model.parameters()).device
    image_dev = image.to(dev)
    label_dev = label.to(dev)

    # Compute clean predictions
    with torch.no_grad():
        clean_probs = F.softmax(model(image_dev.unsqueeze(0)), dim=1).squeeze(0)
        clean_pred = int(clean_probs.argmax().item())

    # Generate adversarial example
    x_adv_dev = make_adv(model, image_dev.unsqueeze(0), label_dev.unsqueeze(0)).squeeze(0)
    perturbation_dev = x_adv_dev - image_dev

    # Compute adversarial predictions
    with torch.no_grad():
        adv_probs = F.softmax(model(x_adv_dev.unsqueeze(0)), dim=1).squeeze(0)
        adv_pred = int(adv_probs.argmax().item())

    # Denormalize for visualization
    image_vis = mnist_denormalize(image_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    x_adv_vis = mnist_denormalize(x_adv_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    perturbation_vis = (x_adv_vis - image_vis)
```

The `unsqueeze(0)` adds a batch dimension since models expect batched inputs. If `clean_probs[3] = 0.95`, the model is 95% confident in class 3. The denormalization step is necessary: `mnist_denormalize` converts normalized-space images back to `[0,1]` pixel space for visualization. Without this step, normalized values like -0.3 or 2.5 would be incorrectly displayed, producing unrecognizable images. The perturbation is computed in pixel space after denormalization to show the actual visual change.

Next, the figure layout is created with three image panels and one probability bar chart:

Code: python

```python
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10), facecolor=NODE_BLACK)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
```

The original image panel shows the clean input with its prediction:

Code: python

```python
    # Original image panel
    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    if image_vis.shape[0] == 1:
        ax1.imshow(image_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(image_vis.permute(1, 2, 0))
    ax1.set_title(f"Original | class={clean_pred} | p={clean_probs[clean_pred]:.2%}",
                  color=HTB_GREEN, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
```

The adversarial image panel uses red title if misclassified:

Code: python

```python
    # Adversarial image panel
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if x_adv_vis.shape[0] == 1:
        ax2.imshow(x_adv_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(x_adv_vis.permute(1, 2, 0))
    title_color = MALWARE_RED if adv_pred != int(label.item()) else HTB_GREEN
    adv_title = f"Adversarial | class={adv_pred} | p={adv_probs[adv_pred]:.2%}"
    if targeted and target_class is not None:
        adv_title += f" | target={target_class}"
    ax2.set_title(adv_title, color=title_color, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
```

The perturbation panel scales differences for visibility:

Code: python

```python
    # Perturbation panel (scaled for visibility)
    ax3 = fig.add_subplot(gs[0, 2])
    _style_axes(ax3)
    pert_scaled = (perturbation_vis * 10 + 0.5).clamp(0, 1)
    if pert_scaled.shape[0] == 1:
        ax3.imshow(pert_scaled.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax3.imshow(pert_scaled.permute(1, 2, 0))
    ax3.set_title("Perturbation (x10)", color=NUGGET_YELLOW, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])
```

The scaling transform `(perturbation_vis * 10 + 0.5)` serves a specific purpose for visualizing signed perturbations. Perturbations can be both positive (brightening pixels) and negative (darkening pixels), typically in a small range like `[-0.05, +0.05]`. Without scaling, these tiny values would appear as uniform gray when displayed. The multiplication by `10` amplifies the perturbations to `[-0.5, +0.5]`, making them visible. The addition of `0.5` centers this range to `[0.0, 1.0]`, where `0.5` represents no change (medium gray), values above `0.5` show positive perturbations (lighter), and values below `0.5` show negative perturbations (darker). For example, a perturbation of `+0.03` becomes `0.03 * 10 + 0.5 = 0.8` (light gray), while `-0.03` becomes `-0.03 * 10 + 0.5 = 0.2` (dark gray). The final `clamp(0, 1)` ensures extreme values stay within the valid display range. The scaling factor `10` is chosen empirically to make typical FGSM perturbations visible without saturating the display.

Finally, the probability comparison bar chart:

Code: python

```python
    # Class probability comparison
    ax4 = fig.add_subplot(gs[1, :])
    _style_axes(ax4)
    x = np.arange(num_classes)
    width = 0.4
    ax4.bar(x - width/2, clean_probs[:num_classes].cpu(), width,
            color=AZURE, label="clean")
    ax4.bar(x + width/2, adv_probs[:num_classes].cpu(), width,
            color=MALWARE_RED, label="adv")
    ax4.set_xlabel("Class", color=WHITE)
    ax4.set_ylabel("Probability", color=WHITE)
    legend = ax4.legend(facecolor=NODE_BLACK, edgecolor=HACKER_GREY)
    for text in legend.get_texts():
        text.set_color(WHITE)
    ax4.set_title("Class probabilities", color=HTB_GREEN, fontweight="bold")
    for text in ax4.get_xticklabels() + ax4.get_yticklabels():
        text.set_color(HACKER_GREY)

    # Add main title and display
    fig.suptitle(title, color=HTB_GREEN, fontweight="bold", fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
```

## FGSM-Specific Wrapper

A thin wrapper simplifies FGSM visualization:

Code: python

```python
def visualize_fgsm_attack(model: nn.Module,
                         image: Tensor,
                         label: Tensor,
                         epsilon: float,
                         num_classes: int = 10,
                         targeted: bool = False,
                         target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using FGSM.

    Args:
        model: Classifier model
        image: Single image tensor
        label: True label
        epsilon: Perturbation budget
        num_classes: Classes to display
        targeted: If True, targeted attack
        target_class: Target class for targeted attack
    """
    def _make_adv(m, xb, yb):
        if targeted and target_class is None:
            raise ValueError("target_class must be provided when targeted=True")
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return fgsm_attack(m, xb, y_used, epsilon, targeted=targeted)

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(model, image, label, _make_adv,
                    title=f"FGSM {mode}",
                    num_classes=num_classes,
                    targeted=targeted,
                    target_class=target_class)
```

The panels show us how a small L∞-bounded change materially changes predicted probabilities. The perturbation view is scaled to enhance visibility and does not reflect true magnitude.

![FGSM untargeted example transforms a digit 7 into class 3 with an amplified perturbation map and probability bars reflecting the misclassification.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/fgsm_untargeted.png)

Rendering a complete view for a single sample:

Code: python

```python
# Assume images, labels from test_loader (from Setup)
# Assume epsilon from Core Implementation (epsilon=0.8)
_ = visualize_fgsm_attack(model, images[0].detach().cpu(),
                         labels[0].detach().cpu(), epsilon)
```

