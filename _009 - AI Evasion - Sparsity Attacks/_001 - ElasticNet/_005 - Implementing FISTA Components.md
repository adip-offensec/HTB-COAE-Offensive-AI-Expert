
# Implementing FISTA Components

The mathematical foundations from the previous section provide the theoretical framework for FISTA optimization. Now we translate that theory into working code. The functional programming approach decomposes the algorithm into small, focused functions that each implement one specific mathematical operation. This section develops the first two building blocks: computing Nesterov momentum and applying soft thresholding.

These functions embody core concepts from convex optimization. The momentum computation implements Nesterov acceleration, which achieves the optimal O(1/k2) rate for smooth convex problems and is used heuristically here. The shrinkage operation implements the proximal operator for the L1 norm, enabling efficient optimization of non-smooth objectives. Together, these pieces form the foundation upon which the complete FISTA iteration will be built.

## Nesterov Momentum: Looking Ahead

The Nesterov acceleration formula tk=k/(k+3) provides momentum that grows with iteration count. Early iterations use small momentum coefficients, taking conservative steps while the optimization explores the loss landscape. Later iterations employ larger coefficients, confidently accelerating along directions that have proven productive across multiple steps.

This specific formula represents a simplified version of the original Nesterov scheme. The full FISTA momentum update uses tk+1=(1+1+4tk2)/2, which approaches k/2 asymptotically. The approximation k/(k+3) provides similar behavior while being slightly more conservative, improving numerical stability without significantly impacting convergence speed.

We implement this as a pure function taking only the iteration number:

Code: python

```python
def compute_fista_momentum(iteration):
    """
    Calculate FISTA momentum parameter for iteration k.

    Uses Nesterov acceleration: k/(k+3)
    Early iterations have small momentum, later ones accelerate.

    Parameters:
        iteration (int): Current FISTA iteration number

    Returns:
        float: Momentum coefficient in [0, 1)
    """
    return iteration / (iteration + 3.0)
```

Don’t let the simplicity fool you. This single line encodes a sophisticated acceleration strategy proven to achieve optimal convergence rates. Why divide by `iteration + 3.0` instead of just using the iteration count? The addition in the denominator keeps the result bounded below 1 (approaching 1 as k→∞), preventing momentum from overwhelming the gradient signal. Without this normalization, momentum could grow unbounded and destabilize optimization entirely.

Let's verify the momentum progression across typical iteration counts:

Code: python

```python
# Test momentum progression
print("FISTA Momentum Progression:")
print("=" * 40)
test_iterations = [1, 5, 10, 50, 100, 500, 1000]

for k in test_iterations:
    momentum = compute_fista_momentum(k)
    print(f"Iteration {k:4d}: momentum = {momentum:.4f}")
```

Expected output:

Code: txt

```txt
FISTA Momentum Progression:
========================================
Iteration    1: momentum = 0.2500
Iteration    5: momentum = 0.6250
Iteration   10: momentum = 0.7692
Iteration   50: momentum = 0.9434
Iteration  100: momentum = 0.9709
Iteration  500: momentum = 0.9940
Iteration 1000: momentum = 0.9970
```

Notice how momentum grows from cautious to aggressive. The first iteration uses only 0.25, providing modest acceleration while letting the gradient dominate direction selection. Why start so conservatively? Early iterations know nothing about the objective landscape, so committing heavily to any direction risks wasting computation. By iteration 10, momentum has grown to 0.77, substantially boosting speed along directions that have proven productive. After 100 iterations, momentum approaches 0.97, providing strong acceleration for the final convergence phase where the algorithm has accumulated substantial evidence about effective search directions.

This growth pattern serves a specific purpose. When optimization begins, the algorithm knows little about the objective landscape. Small momentum prevents premature commitment to directions that might prove unproductive. As iterations accumulate evidence that certain directions consistently reduce the objective, larger momentum exploits this structure for faster convergence.

Contrast this with standard momentum methods that use a fixed coefficient (often 0.9 throughout training). Fixed momentum requires careful tuning: too small and convergence is slow, too large and optimization becomes unstable. Nesterov's adaptive schedule eliminates this hyperparameter, automatically adjusting acceleration to match the optimization phase.

## Soft Thresholding: Creating Sparsity

The shrinkage-thresholding operation implements the proximal operator for the L1 norm developed in Proximal Operators and Soft Thresholding. This function takes candidate adversarial images y, applies the three-region thresholding rule (positive shrinkage, dead zone elimination, negative shrinkage), and returns sparse perturbations.

Code: python

```python
def apply_shrinkage_thresholding(y, original_images, threshold, clip_min=0.0, clip_max=1.0):
    """
    Apply soft thresholding to create sparse perturbations.

    Values with magnitude at most threshold are zeroed (sparsity).
    Values exceeding the threshold are reduced by the threshold but remain non-zero.

    Parameters:
        y (torch.Tensor): Candidate adversarial images after gradient step
        original_images (torch.Tensor): Original unperturbed images
        threshold (float): Soft thresholding parameter (higher = sparser)
        clip_min (float): Minimum valid pixel value (default 0.0)
        clip_max (float): Maximum valid pixel value (default 1.0)

    Returns:
        torch.Tensor: Images after soft thresholding
    """
    # Compute the difference from original
    diff = y - original_images

    # Apply soft thresholding
    # Values within threshold are zeroed (sparsity!)
    # Values above threshold get reduced by threshold
    shrink_positive = torch.clamp(y - threshold, min=clip_min, max=clip_max)
    shrink_negative = torch.clamp(y + threshold, min=clip_min, max=clip_max)

    # Three-way decision: positive, zero, or negative
    cond_positive = (diff > threshold).float()
    cond_zero = (torch.abs(diff) <= threshold).float()
    cond_negative = (diff < -threshold).float()

    # Combine using conditions
    result = (
        cond_positive * shrink_positive
        + cond_zero * original_images
        + cond_negative * shrink_negative
    )

    return result
```

Computing `diff = y - original_images` reveals the perturbation that gradient descent would produce before thresholding. This combines the smooth gradient step (from adversarial loss and L2 penalty) with previous momentum. For a specific pixel, if the original value was 0.5 and the candidate adversarial value `y` is 0.62, then `diff = 0.12`, indicating a positive perturbation. Soft thresholding will examine this 0.12 against threshold β to decide whether it survives or gets eliminated.

Preparing the shrunk values happens next through two separate computations. For positive perturbations, `shrink_positive = torch.clamp(y - threshold, min=clip_min, max=clip_max)` subtracts β from the candidate values. With a pixel at 0.62 and β=0.1, we get `0.62 - 0.1 = 0.52` as the "shrunk" value that will be used if this perturbation survives. Clamping at `clip_min` prevents results from dropping below 0.0 (the minimum valid pixel value).

Negative perturbations get handled symmetrically. `shrink_negative = torch.clamp(y + threshold, ...)` adds β instead of subtracting, while clamping at `clip_max` prevents values from exceeding 1.0. These shrunk values sit ready but unused until the masking step decides which pixels actually need them.

Applying the three-way decision requires binary masks for each region. `cond_positive`, `cond_zero`, and `cond_negative` each produce tensors of 0s and 1s indicating which pixels fall into that threshold category. Multiplying these masks element-wise with their corresponding values (`shrink_positive`, `original_images`, `shrink_negative`) and summing creates the final result. For pixels with perturbations exceeding β, the positive mask selects the shrunk value. For pixels in the dead zone, the zero mask reverts to the original. For negative perturbations below −β, the negative mask applies negative shrinkage.

## Visualizing Shrinkage Effects

To understand how soft thresholding creates sparsity, let's apply it to sample perturbations and visualize the results. We'll create synthetic perturbations with controlled properties and observe how thresholding modifies them:

Code: python

```python
# Create synthetic perturbation for visualization
print("\nTesting shrinkage-thresholding operation...")

# Generate a 5x5 synthetic perturbation pattern
perturbation_pattern = torch.tensor([
    [-0.15, -0.08, -0.02,  0.03,  0.12],
    [-0.10, -0.05,  0.00,  0.06,  0.18],
    [-0.05,  0.00,  0.05,  0.10,  0.20],
    [ 0.00,  0.05,  0.08,  0.15,  0.25],
    [ 0.05,  0.08,  0.12,  0.20,  0.30]
], device=device)

# Assume original pixels are 0.5 (mid-gray)
original_values = torch.ones_like(perturbation_pattern) * 0.5
perturbed_values = original_values + perturbation_pattern

# Apply shrinkage with beta=0.1
beta_test = 0.1
thresholded = apply_shrinkage_thresholding(
    perturbed_values,
    original_values,
    beta_test,
    clip_min=0.0,
    clip_max=1.0
)

# Compute resulting perturbations
resulting_perturbation = thresholded - original_values

print(f"\nSoft Thresholding Results (beta={beta_test}):")
print("=" * 50)
print("\nOriginal perturbation pattern:")
print(perturbation_pattern.cpu().numpy())
print("\nAfter soft thresholding:")
print(resulting_perturbation.cpu().numpy())

# Count sparsity
original_nonzero = (perturbation_pattern.abs() > 1e-6).sum().item()
thresholded_nonzero = (resulting_perturbation.abs() > 1e-6).sum().item()
sparsity_gained = original_nonzero - thresholded_nonzero

print(f"\nSparsity Analysis:")
print(f"  Original non-zero elements: {original_nonzero}/25")
print(f"  After thresholding: {thresholded_nonzero}/25")
print(f"  Elements zeroed out: {sparsity_gained}")
print(f"  Sparsity achieved: {(25-thresholded_nonzero)/25*100:.1f}%")
```

Expected output:

Code: txt

```txt
Testing shrinkage-thresholding operation...

Soft Thresholding Results (beta=0.1):
==================================================

Original perturbation pattern:
[[-0.15 -0.08 -0.02  0.03  0.12]
 [-0.1  -0.05  0.    0.06  0.18]
 [-0.05  0.    0.05  0.1   0.2 ]
 [ 0.    0.05  0.08  0.15  0.25]
 [ 0.05  0.08  0.12  0.2   0.3 ]]

After soft thresholding:
[[-0.05  0.    0.    0.    0.02]
 [ 0.    0.    0.    0.    0.08]
 [ 0.    0.    0.    0.    0.1 ]
 [ 0.    0.    0.    0.05  0.15]
 [ 0.    0.    0.02  0.1   0.2 ]]

Sparsity Analysis:
  Original non-zero elements: 22/25
  After thresholding: 9/25
  Elements zeroed out: 13
  Sparsity achieved: 64.0%
```

Sparsity emerges through magnitude-based pruning. Before thresholding, 22 of 25 elements had non-zero perturbations (88% dense). After applying β=0.1, only 9 survived (64% sparse). Crucially, eliminated elements become exact zeros, not small values like 0.001 that would technically count as non-zero in the L0 norm.

Why does elimination follow a spatial pattern? The original perturbation was structured with small values in the upper-left (range -0.15 to 0.03) and large values in the bottom-right (range 0.12 to 0.30). Setting β=0.1 creates a magnitude threshold that bisects this distribution. The upper-left region falls entirely below threshold: every value satisfies |p|≤0.1, triggering complete elimination. The central diagonal, with values hovering near 0.1, also vanishes. Only the bottom-right corner exceeds threshold, though each surviving value gets shrunk by exactly β=0.1. For example, the original 0.30 perturbation becomes 0.30−0.1=0.20 after shrinkage.

This magnitude-based filtering serves adversarial attacks perfectly. Pixels with weak gradients produce small perturbations that waste L0 budget without meaningfully shifting the decision boundary. Thresholding discards these automatically, concentrating the budget on high-gradient regions where perturbations actually change predictions. The threshold β controls the severity of pruning: β=0.2 would eliminate 20 of 25 elements (80% sparse), preserving only the extreme values, while β=0.05 would keep 14 elements (44% sparse), allowing more moderate perturbations to contribute.

The next section examines how gradients flow through the model during backpropagation. Understanding gradient magnitudes, loss landscape properties, and convergence behavior provides insight into FISTA's optimization dynamics and helps diagnose potential issues during attack execution.

