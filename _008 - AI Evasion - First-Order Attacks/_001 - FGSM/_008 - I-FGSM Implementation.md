
Building on Kurakin et al.’s I-FGSM algorithm (2016), the implementation of the projected iterative update uses α=ϵT by default with optional targeted behavior. Each iteration recomputes the gradient at the current adversarial and applies the per-pixel sign step followed by projection. The projection line enforces the L∞ cap relative to the original input rather than the previous iterate, which keeps the budget interpretable as a maximum per-pixel change from x.

## Core Iterative Algorithm

What makes iteration work? Small steps with fresh gradients. At each iteration, compute where you are, find the steepest direction, take a small sign step, then snap back to the allowed region:

Code: python

```python
def iterative_fgsm(model: nn.Module,
                   images: Tensor,
                   labels: Tensor,
                   epsilon: float,
                   num_iter: int,
                   alpha: float | None = None,
                   targeted: bool = False,
                   random_start: bool = False) -> Tensor:
    """Iterative FGSM (Basic Iterative Method) with projection.

    Args:
        model: Target classifier in evaluation mode
        images: Clean images (normalized)
        labels: Ground-truth or target labels
        epsilon: L_infinity budget (in normalized space)
        num_iter: Number of iterations
        alpha: Step size per iteration (defaults to epsilon/T)
        targeted: If True, targeted attack
        random_start: If True, initialize within the epsilon ball

    Returns:
        Tensor: Adversarial images (normalized)
    """
    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if alpha is None:
        alpha = epsilon / max(num_iter, 1)
    if random_start:
        torch.manual_seed(1337)
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(images + delta, MNIST_NORM_MIN, MNIST_NORM_MAX)
    else:
        x_adv = images.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step_dir = -1.0 if targeted else 1.0
        x_adv = x_adv + step_dir * alpha * x_adv.grad.sign()
        x_adv = torch.clamp(images + (x_adv - images).clamp(-epsilon, epsilon), MNIST_NORM_MIN, MNIST_NORM_MAX)

    return x_adv.detach()
```

The default step size `alpha = epsilon / max(num_iter, 1)` divides the budget across iterations. If `epsilon=0.8` and `num_iter=10`, then `alpha=0.08` per step. Even if all 10 steps point in the same direction (unlikely), the total movement is capped at 10×0.08=0.8. The random start option adds uniform noise in `[-epsilon, epsilon]` before iteration begins, helping escape local neighborhoods where the gradient points nowhere useful.

How does projection keep the perturbation honest? The line `(x_adv - images).clamp(-epsilon, epsilon)` measures how far each pixel has drifted from its original value and clips any drift exceeding ±ϵ. If a pixel has moved +0.9 but `epsilon=0.8`, clamp reduces it to +0.8. The outer `torch.clamp(..., MNIST_NORM_MIN, MNIST_NORM_MAX)` then ensures all pixel values stay in the valid normalized input range. Projection happens relative to the original `images`, not the previous iterate, so the budget always means "ϵ away from the starting point," not "ϵ per step."

## Testing Iterative FGSM

Does iteration really improve over single-step FGSM? Let’s run I-FGSM on a batch and measure the flip rate. The test uses α=ϵT and random starts to maximize effectiveness:

Code: python

```python
# Assume model, test_loader, device from FGSM Setup
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

epsilon = 0.8
num_iter = 10
alpha = epsilon / num_iter  # alpha = 0.08

with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon=epsilon,
    num_iter=num_iter,
    alpha=alpha,
    targeted=False,
    random_start=True
)

with torch.no_grad():
    adv_pred_ifgsm = model(x_adv_ifgsm).argmax(dim=1)

originally_correct = clean_pred == labels
flipped_ifgsm = (adv_pred_ifgsm != labels) & originally_correct
print(
    f"I-FGSM flips (first batch): "
    f"{(flipped_ifgsm.float().sum() / originally_correct.float().sum().clamp_min(1.0)).item():.2%}"
)
```

Expected output:

Code: txt

```txt
I-FGSM flips (first batch): 100.00%
```

The printed flip rate focuses on samples the model originally classified correctly to isolate genuine attack impact. In this run, every originally correct sample flipped, yielding 100.00%. Compare this to FGSM's 71.09% at the same epsilon. Iteration improves the success rate without increasing the perturbation budget.

## Measuring Attack Impact

The flip rate is much higher, but what else changed? Confidence levels, perturbation norms, and the distribution of attack success across samples all provide insight into how α and T affect outcomes:

Code: python

```python
# Reuse evaluate_attack function from the Evaluation Metrics section
metrics_ifgsm = evaluate_attack(model, images, x_adv_ifgsm, labels)
for k, v in metrics_ifgsm.items():
    print(f"{k}: {v:.4f}")
```

Expected output:

Code: txt

```txt
clean_accuracy: 1.0000
adversarial_accuracy: 0.0000
attack_success_rate: 1.0000
avg_clean_confidence: 0.9853
avg_adv_confidence: 0.0115
avg_confidence_drop: 0.9739
avg_l2_perturbation: 14.0519
max_linf_perturbation: 0.8000
```

These results show a complete collapse in accuracy under I-FGSM at the tested budget. With `clean_accuracy` of 1.0000, all samples were initially correct. After the attack, `adversarial_accuracy` drops to 0.0000 and `attack_success_rate` reaches 1.0000, meaning every originally correct sample was flipped. The `avg_confidence_drop` of 0.9739 indicates the model’s probability assigned to the true class fell by about 97 percentage points on average, from 0.9853 to 0.0115. The `max_linf_perturbation` confirms the ϵ=0.8 bound is respected; the higher `avg_l2_perturbation` reflects that the budget is used broadly across pixels rather than concentrated in a few locations.

Finally, the `max_linf_perturbation` of 0.8000 confirms the attack respects the epsilon bound in normalized space. Increasing T with α=ϵT often raises success rate without increasing max⁡∥δ∥∞. The L2 norm may drop slightly as the attack distributes the budget more efficiently across pixels rather than overshooting in some dimensions.

