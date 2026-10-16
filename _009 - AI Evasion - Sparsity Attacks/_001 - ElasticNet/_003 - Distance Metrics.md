
# Distance Metrics

To balance sparsity against smoothness, ElasticNet needs three distance computations: L1 for counting total change, squared L2 for measuring energy, and their weighted combination for the actual optimization objective. Computing these efficiently on batched tensors while preserving per-example metrics requires careful dimension handling.

## Measuring Perturbation Magnitude

The L1 distance tells us the total magnitude of change summed across all pixels. For a 28×28 MNIST image with 784 pixels, this sum ranges from 0 (no change) to potentially hundreds (many pixels changed substantially). If 100 pixels each change by 0.15 units, the L1 distance is 100×0.15=15.0. If instead 50 pixels each change by 0.30 units, the L1 distance is still 50×0.30=15.0. Unlike L0 which counts changed pixels (100 versus 50), L1 weights each pixel by how much it changed.

The L2 distance measures the squared straight-line distance in 784-dimensional pixel space. Squaring individual perturbations before summing creates a strong penalty for large individual changes. Consider the same two perturbations: 100 pixels at 0.15 each yields L2=100×0.152=2.25, while 50 pixels at 0.30 each yields L2=50×0.302=4.50. The second perturbation has twice the L2 distance despite identical L1 distance, because L2 penalizes concentrated changes more heavily than distributed changes.

The elastic-net distance ∥δ∥22+β∥δ∥1 combines both norms with parameter β controlling their relative weight. When β=0, we recover pure L2 distance. As β increases, the L1 component gains influence, biasing optimization toward sparser solutions. This mixed distance lacks the clean geometric interpretation of pure norms, but it captures the sparsity-smoothness balance that pure norms cannot.

## Understanding Norm Trade-offs

Different norms encode different notions of perturbation quality. Pure L2 optimization produces smooth, distributed perturbations where many pixels change by small amounts. This distributes the perturbation budget evenly, avoiding concentrated modifications that might be visually obvious. However, L2 alone provides no sparsity guarantee. An L2-optimal perturbation might modify every single pixel by a tiny amount, achieving low Euclidean distance while creating perceptually noticeable global shifts in brightness or texture.

Pure L1 optimization encourages sparsity through its non-differentiable kink at zero. The absolute value function has no unique derivative at the origin, creating a "barrier" that optimization must overcome to activate a pixel. This barrier preferentially keeps small perturbations at exactly zero, concentrating the perturbation budget into fewer pixels with larger individual changes. For a fixed L1 budget, modifying 50 pixels at 0.2 each gives the same L1 distance as modifying 100 pixels at 0.1 each, but the sparse solution (50 pixels) often proves more effective at fooling classifiers because concentrated changes can cross local decision boundaries more decisively.

The elastic-net combination provides a middle ground. The L2 term prevents any single pixel from dominating the perturbation (since squaring penalizes large values quadratically), while the L1 term encourages many pixels to remain at exactly zero. Tuning β controls this balance. Small β (like 0.01) produces nearly-smooth perturbations with mild sparsity, suitable when imperceptibility matters more than pixel count. Large β (like 1.0) produces aggressive sparsity, potentially concentrating perturbations into visually obvious clusters but minimizing the number of modified pixels.

Code: python

```python
def compute_distances(adv_images, original_images, beta):
    """
    Compute L1, L2, and elastic-net distances.

    Returns all three distance metrics used in optimization
    and decision-making.

    Parameters:
        adv_images (torch.Tensor): Adversarial images (batch_size, C, H, W)
        original_images (torch.Tensor): Original images (batch_size, C, H, W)
        beta (float): Weight for L1 in elastic-net distance

    Returns:
        tuple: (l1_dist, l2_dist, elastic_dist) each shape (batch_size,)
    """
    l1_dist = torch.sum(torch.abs(adv_images - original_images), dim=(1, 2, 3))
    l2_dist = torch.sum((adv_images - original_images) ** 2, dim=(1, 2, 3))
    elastic_dist = l2_dist + beta * l1_dist

    return l1_dist, l2_dist, elastic_dist
```

The `dim=(1, 2, 3)` parameter collapses channel, height, and width while preserving the batch dimension. For a batch of 20 images with shape (20, 1, 28, 28), this produces 20 scalar distances - one per image. Why preserve the batch dimension? Binary search adapts trade-off constants individually: an easy example near the decision boundary might use c=0.01 while a resistant example deep in the correct class region needs c=10. Without per-example distances, we’d be forced to use the same constant for all images, over-perturbing easy cases or under-perturbing hard ones.

Notice we’re using squared L2 distance ∥x′−x∥22 for the same computational reasons discussed in the FISTA framework: the gradient 2(x′−x) is simpler than the normalized form from the true L2 norm. This maintains strict convexity and produces the same optimal perturbations. The elastic distance combines the squared L2 component with the L1 term, using the same β weighting that appears in our proximal operator.

Note what this function does not do: it doesn't apply any constraints, perform any thresholding, or make any decisions. It simply measures. Other functions will use these distances to make decisions about attack success or to compute loss gradients, but the distance computation itself remains a pure measurement operation.

With distance metrics defined, the next section implements the adversarial loss functions that drive misclassification and combines them with these distance terms to form the complete optimization objective.

