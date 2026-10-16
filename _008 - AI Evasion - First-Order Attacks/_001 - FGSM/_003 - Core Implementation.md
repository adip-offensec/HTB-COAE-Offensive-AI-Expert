
# Core Implementation

Before building the attack, we need a clean foundation. We assume you have the required libraries imported, a trained `SimpleCNN`, a `test_loader`, and a configured `device` from the setup section. The implementation uses PyTorch for tensor operations and automatic differentiation, along with the visualization tools and HTB styling constants already available from previous sections.

## Computing Loss Without Side Effects

Separating forward computation and loss calculation into a small helper keeps gradient flow scoped to the input tensor and makes debugging simpler. This helper assumes the model is already in evaluation mode to avoid state updates from batch normalization or dropout. Keeping the function free of graph‑modifying side effects lets us reuse it wherever a clean loss value is needed:

Code: python

```python
def _forward_and_loss(model: nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Forward pass and cross-entropy loss without side effects.

    Args:
        model: Neural network classifier
        x: Input images tensor
        y: Target labels tensor

    Returns:
        tuple[Tensor, Tensor]: Model logits and scalar loss value
    """
    if getattr(model, "training", False):
        raise RuntimeError("Expected model.eval() for attack computations to avoid BN/Dropout state updates")
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return logits, loss
```

What do the shapes look like? For a batch of 128 MNIST images, `x` arrives with shape `[128, 1, 28, 28]` (batch size, channels, height, width). The model outputs `logits` of shape `[128, 10]`, one row per image with 10 class scores. Cross-entropy reduces these to a single scalar `loss` value. Returning both logits and loss gives callers flexibility: sometimes you need predictions, sometimes just the loss, sometimes both.

## Gradient Computation

As established in the FGSM theory section, adversarial attacks require computing gradients with respect to inputs rather than parameters. Here's the PyTorch implementation:

Code: python

```python
def _input_gradient(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Return gradient of loss with respect to input tensor x.

    Args:
        model: Neural network in evaluation mode
        x: Input images to compute gradients for
        y: True labels for loss computation

    Returns:
        Tensor: Gradient tensor with same shape as x
    """
    x_req = x.clone().detach().requires_grad_(True)
    _, loss = _forward_and_loss(model, x_req, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return x_req.grad.detach()
```

The implementation details: `x.clone().detach()` creates a new tensor disconnected from the computational history. The `.requires_grad_(True)` enables gradient tracking on the input. When `loss.backward()` executes, PyTorch computes ∇xℒ and stores it in `x_req.grad`. The final `.detach()` returns a standalone gradient tensor. Shape is preserved automatically: a `[128, 1, 28, 28]` input produces a `[128, 1, 28, 28]` gradient.

## Core FGSM Attack Implementation

With gradients in hand, the attack becomes almost trivial. Extract the sign, scale by ϵ, add to the image, and clip. Four operations implement the entire attack:

Code: python

```python
def fgsm_attack(model: nn.Module,
                images: Tensor,
                labels: Tensor,
                epsilon: float,
                targeted: bool = False) -> Tensor:

    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not images.is_floating_point():
        raise ValueError("images must be floating point tensors")

    grad = _input_gradient(model, images, labels)
    step_dir = -1.0 if targeted else 1.0
    x_adv = images + step_dir * epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, MNIST_NORM_MIN, MNIST_NORM_MAX)
    return x_adv.detach()
```

PyTorch’s `.sign()` method operates element-wise on tensors, preserving shape and device placement. For a gradient tensor of shape `[128, 1, 28, 28]` on GPU, `grad.sign()` returns another `[128, 1, 28, 28]` tensor on the same GPU with identical memory layout. The operation is computationally cheap (single pass, no sorting or reduction) and memory-efficient (no intermediate allocations beyond the output tensor). The multiplication `epsilon * grad.sign()` broadcasts the scalar `epsilon` across all tensor elements, creating the perturbation δ where each component has magnitude exactly `epsilon` (or zero if the gradient was exactly zero, which rarely occurs with floating-point arithmetic).

The scalar `step_dir` controls attack direction through multiplication. Setting `step_dir=1.0` for untargeted attacks means the perturbation `step_dir * epsilon * grad.sign()` adds to the image, while `step_dir=-1.0` for targeted attacks subtracts. This avoids duplicating the gradient computation logic. The `torch.clamp()` operation clips adversarial images to the valid normalized range for MNIST (approximately -0.424 to 2.821), preventing grey washout when denormalized for visualization. This range corresponds to [0,1] in pixel space after applying the inverse normalization transform. Clamping happens element-wise, and the final `.detach()` extracts the adversarial images without gradient tracking.

## Testing the Attack

Does FGSM actually work? Let's grab a batch, attack it, and count how many predictions flip. The test focuses on samples the model originally classified correctly, since flipping an already-wrong prediction tells us nothing:

Code: python

```python
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
# Epsilon in normalized space (≈0.25 in pixel space)
epsilon = 0.8
with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv = fgsm_attack(model, images, labels, epsilon)
with torch.no_grad():
    adv_pred = model(x_adv).argmax(dim=1)

originally_correct = (clean_pred == labels)
flipped = (adv_pred != labels) & originally_correct
success = flipped.sum().item() / max(int(originally_correct.sum().item()), 1)
print(f"FGSM flips (first batch): {success:.2%}")
```

Expected output:

Code: txt

```txt
FGSM flips (first batch): 71.09%
```

How should we interpret a 71.09% flip rate? Out of 128 samples in the batch, if 128 were originally correct and 91 flipped, the success rate would be 91/128≈0.7109. This run uses ϵ=0.8 in normalized MNIST space. With mean 0.1307 and std 0.3081, this maps to approximately 0.8×0.3081≈0.25 in [0,1] pixel space, about 0.25×255≈64 intensity levels at 8-bit precision. The exact flip rate varies with the trained weights and the specific batch.

## Pixel-Space FGSM Variant

The core FGSM implementation works entirely in normalized space: it expects images already normalized, uses epsilon in normalized units, and returns adversarials in normalized space. This matches our setup where `get_mnist_loaders(normalize=True)` provides pre-normalized batches. However, some workflows start with raw pixel-space images in `[0,1]` and need epsilon specified as pixel-space perturbations (like "8/255 intensity levels").

For these pixel-space workflows attacking normalized models, we need a variant that accepts `[0,1]` images, interprets ϵ as a pixel-space budget, but still normalizes internally for the model and converts gradients correctly. The key is dividing normalized-space gradients by σ to map them back to pixel space before applying the sign step with the pixel-space ϵ.

Why the conversion? When gradients are computed with respect to normalized inputs xnorm=(x−μ)/σ, those gradients live in normalized space. To apply a pixel-space epsilon budget, we must convert gradients back using the chain rule: ∇xℒ=∇xnormℒ/σ. This ensures `epsilon=0.031` (approximately 8/255) means exactly 8 intensity levels in original `[0,1]` space, not in the stretched normalized space where magnitudes differ by a factor of σ.

### Creating Normalization Parameters

Before implementing the pixel-space FGSM variant, we need a helper function to prepare normalization parameters as broadcastable tensors matching image batch dimensions. The normalization transform requires mean and standard deviation values, but we receive them as Python lists. Converting them to properly shaped tensors ensures they broadcast correctly across batch, height, and width dimensions when normalizing images:

Code: python

```python
def _norm_params(images: Tensor, mean: list, std: list) -> tuple[Tensor, Tensor]:
    """Convert normalization parameters to broadcastable tensors.

    Args:
        images: Input images tensor with shape (N, C, H, W)
        mean: Normalization mean per channel as list
        std: Normalization std per channel as list

    Returns:
        tuple[Tensor, Tensor]: Mean and std tensors with shape (1, C, 1, 1)
    """
    device, dtype, C = images.device, images.dtype, images.shape[1]
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    if mean_t.shape[1] != C or std_t.shape[1] != C:
        raise ValueError("mean/std channels must match images")
    return mean_t, std_t
```

The `.view(1, -1, 1, 1)` reshapes the mean/std vectors for broadcasting. For RGB images with `mean=[0.485, 0.456, 0.406]`, the tensor shape becomes `[1, 3, 1, 1]`, broadcasting across batch, height, and width dimensions.

### Pixel-Space FGSM Implementation

How do we bridge pixel-space inputs with normalized models while keeping epsilon interpretable? The implementation needs to handle a tricky dance: accept `[0,1]` images, transform them to normalized space for the model’s gradient computation, then transform those gradients back to pixel space so our pixel-space epsilon has correct meaning. The key challenge is the gradient conversion step, where we must divide by σ to undo the scaling that normalization introduced. Without this step, applying a pixel-space epsilon to normalized-space gradients produces incorrect perturbation magnitudes.

Code: python

```python
def fgsm_pixel_space(model: nn.Module,
                     images: Tensor,
                     labels: Tensor,
                     epsilon: float,
                     mean: list,
                     std: list,
                     targeted: bool = False) -> Tensor:
    """FGSM for pixel-space inputs attacking normalized models.

    This variant accepts images in [0,1] pixel space rather than normalized
    space. It normalizes inputs internally for the model, converts gradients
    back to pixel space, and returns adversarials in [0,1] pixel space.

    Args:
        model: Model expecting normalized inputs
        images: Clean images in [0,1] pixel space (unnormalized)
        labels: Target labels
        epsilon: Max perturbation in pixel space (e.g., 8/255)
        mean: Normalization mean per channel
        std: Normalization std per channel
        targeted: If True, minimize loss towards labels

    Returns:
        Tensor: Adversarial images in [0,1] pixel space (unnormalized)
    """
    mean_t, std_t = _norm_params(images, mean, std)
    x = images.clone().detach()
    x_norm = (x - mean_t) / std_t
    x_norm.requires_grad_(True)

    _, loss = _forward_and_loss(model, x_norm, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Convert gradient from normalized space to image space
    grad_img = x_norm.grad / std_t
    step_dir = -1.0 if targeted else 1.0
    x_adv = torch.clamp(x + step_dir * epsilon * grad_img.sign(), 0.0, 1.0)
    return x_adv.detach()
```

Why divide by `std_t`? The normalization (x−μ)/σ scales pixel values. Gradients computed with respect to the normalized input xnorm live in that scaled space. To interpret `epsilon` as a bound in the original [0,1] pixel space, we must undo the scaling. If `std=[0.229]` and the normalized-space gradient is 2.5, the image-space gradient becomes 2.5/0.229≈10.9. The sign step then uses this rescaled gradient, ensuring ϵ=8/255≈0.031 means exactly 8 intensity levels in the original image, not 8 levels in the normalized space where magnitudes differ.

### When to Use This Variant

This pixel-space variant is useful when working with pipelines that provide raw unnormalized images but you need to attack a model expecting normalized inputs. For example, some datasets or external APIs return images in `[0,1]` without normalization applied.

However, if your data is already normalized (as with our `get_mnist_loaders(normalize=True)` setup), you should use the core `fgsm_attack` directly, which operates entirely in normalized space. Converting normalized → pixel → attack → normalized is unnecessarily complex.

For workflows requiring this variant: if images arrive in `[0,1]`, specify ϵ in pixel units (like 8/255≈0.031), and the function handles the internal normalization and gradient conversion automatically:

Code: python

```python
# Example: Starting with pixel-space images
epsilon_px = 8 / 255  # pixel-space epsilon (≈0.031)
mean, std = [0.1307], [0.3081]

# Denormalize existing normalized images to get pixel-space images
mean_t, std_t = _norm_params(images, mean, std)
pixel_images = images * std_t + mean_t
pixel_images = torch.clamp(pixel_images, 0.0, 1.0)

# Attack in pixel space
x_adv_pixel = fgsm_pixel_space(model, pixel_images, labels, epsilon_px, mean, std)

# x_adv_pixel is in [0,1] and can be displayed or saved directly
# If you need to pass to the model again, normalize it first:
x_adv_norm = (x_adv_pixel - mean_t) / std_t
```

The function ensures ϵ retains its pixel-space interpretation throughout the attack by converting gradients via division by σ before applying the sign step. This keeps perturbation budgets meaningful when comparing across different normalization schemes.

