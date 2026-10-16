The theoretical foundation established how DeepFool iteratively linearizes decision boundaries to find minimal perturbations. Now we translate this mathematical framework into executable code that performs multi-class attacks through adaptive boundary detection and refinement.

## Function Definition and Initialization

The DeepFool function accepts an input image and a trained neural network, along with parameters controlling the algorithm's behavior. The `num_classes` parameter limits the search to the top-scoring classes, reducing computational cost. The `overshoot` factor ensures the algorithm crosses the true non-linear boundary rather than just touching the linearized approximation.

Code: python

```python
def deepfool(image: torch.Tensor,
             net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             device: str = 'cuda') -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """
    Generate minimal adversarial perturbation using DeepFool algorithm.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        net (nn.Module): Target neural network in evaluation mode
        num_classes (int): Number of top-scoring classes to consider (default: 10)
        overshoot (float): Overshoot parameter for boundary crossing (default: 0.02)
        max_iter (int): Maximum iterations before terminating (default: 50)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - r_tot (torch.Tensor): Total accumulated perturbation
            - loop_i (int): Number of iterations performed
            - label (int): Original predicted class
            - k_i (int): Final adversarial class
            - pert_image (torch.Tensor): Final perturbed image
    """
    image = image.to(device)
    net = net.to(device)
```

Device transfers ensure computations occur on the selected hardware for both inputs and model. The function signature defines the expected return values: the total perturbation accumulated across iterations, the number of iterations required, the original and adversarial class labels, and the final perturbed image.

## Initial Classification and Class Ranking

The algorithm begins by establishing the original classification and ranking all classes by their scores. This ranking determines the order in which alternative classes are considered during the search for the minimal perturbation.

Code: python

```python
    # Original prediction and class ordering (descending score)
    f_image = net(image).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    label = I[0]
```

The network produces logits for each class, which are converted to a NumPy array and flattened to 1D. The expression `argsort()[::-1]` sorts class indices in descending order of their scores. For example, if the logits are `[2.1, 5.3, 0.8, 3.7]` for classes 0-3, then `I` becomes `[1, 3, 0, 2]`, indicating class 1 has the highest score (5.3), followed by class 3 (3.7), class 0 (2.1), and class 2 (0.8). The variable `label = I[0]` selects the predicted class, which is the index with the highest score.

## Initialization of Working Variables

The algorithm maintains several tensors throughout the iterative process: the current perturbed image, the accumulated perturbation, and iteration counters.

Code: python

```python
    # Working tensors and accumulators
    input_shape = image.shape
    pert_image = image.clone()
    r_tot = torch.zeros(input_shape).to(device)
    loop_i = 0
```

The tensor `pert_image` holds the current perturbed image, initially equal to the original input. The accumulator `r_tot` tracks the total perturbation applied across all iterations, starting at zero. As the algorithm iterates, `r_tot` accumulates the incremental perturbations, ultimately representing the complete adversarial perturbation. The counter `loop_i` tracks iterations to enforce the maximum iteration limit.

## Main Iteration Loop

The algorithm iteratively refines the perturbation until the classification changes or the maximum iteration limit is reached. Each iteration performs a local linearization and takes a step toward the nearest decision boundary.

Code: python

```python
    # Iterate until a successful perturbation is found or the limit is reached
    while loop_i < max_iter:
        x = pert_image.clone().requires_grad_(True)
        fs = net(x)
        # Current top prediction at x
        k_i = fs.data.cpu().numpy().flatten().argsort()[::-1][0]

        # Stop when the prediction changes
        if k_i != label:
            break

        # Initialize the best candidate step for this iteration
        pert = float('inf')
        w = None
```

At each iteration, the classifier is re-evaluated at the current perturbed image `x`. The gradient computation requires `requires_grad_(True)` to enable automatic differentiation. The expression `fs.data.cpu().numpy().flatten().argsort()[::-1][0]` converts the logits to a NumPy array, sorts the classes by score in descending order, and selects the top class. If this predicted class `k_i` differs from the original `label`, the attack has succeeded and the loop terminates.

The variables `pert` and `w` track the smallest distance ratio and its corresponding gradient direction for the current iteration. Setting `pert` to infinity ensures any valid candidate will be smaller, guaranteeing the first valid boundary distance replaces this placeholder.

## Finding the Minimal Step Direction

For each iteration, the algorithm examines the top-scoring alternative classes to find which has the closest decision boundary. This search involves computing gradients for each candidate, measuring distances, and tracking the minimum.

### Setting Up the Candidate Class Loop

Code: python

```python
        # Search minimal step among candidate classes
        for k in range(1, num_classes):
            if I[k] == label:
                continue
```

The loop starts at index 1 (not 0) because `I[0]` is the current predicted class, which we're trying to move away from. The variable `I` contains class indices sorted by descending score from the initial prediction. If `I[k]` matches the current `label`, we skip it since moving toward the current class makes no sense. The `continue` statement immediately advances to the next iteration without executing the gradient computations below.

### Computing Gradient for Candidate Class

Code: python

```python
            # Compute gradient for candidate class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            grad_k = x.grad.data.clone()
```

Before computing a new gradient, we zero any existing gradient accumulated in `x.grad` from previous operations. The `.backward()` call computes gradients of the candidate class score `fs[0, I[k]]` with respect to the input `x`. The expression `fs[0, I[k]]` accesses the logit for class `I[k]` in the single-item batch: index `0` selects the batch dimension, `I[k]` selects the class dimension. The parameter `retain_graph=True` preserves the computation graph for subsequent backward passes. Without this, PyTorch would free the graph memory, causing errors when computing `grad_label` next. The `.clone()` creates an independent copy, preventing it from being overwritten when we zero gradients again.

### Computing Gradient for Original Class

Code: python

```python
            # Compute gradient for original class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, label].backward(retain_graph=True)
            grad_label = x.grad.data.clone()
```

We repeat the gradient computation process for the original class. Zeroing `x.grad` removes the candidate class gradient computed above. The backward pass computes how the original class score changes with input modifications. The `grad_label` tensor points in the direction that maximally increases the original class confidence. By computing both `grad_k` and `grad_label`, we can find the direction that simultaneously increases the candidate class while decreasing the original class.

### Computing Direction and Distance

Code: python

```python
            # Direction and distance under linearization
            w_k = grad_k - grad_label
            f_k = (fs[0, I[k]] - fs[0, label]).data.cpu().numpy()
            pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-10)
```

The difference `w_k = grad_k - grad_label` creates a combined gradient that simultaneously increases the candidate class score while decreasing the original class score. Think of this as a tug-of-war: `grad_k` pulls toward the candidate, `grad_label` pulls toward the original, and `w_k` is the net direction.

The score gap `f_k` represents the deficit to overcome. If the original class has score 8.5 and the candidate has score 3.2, then `f_k = 3.2 - 8.5 = -5.3`, a 5.3-point deficit. The ratio `pert_k = |f_k| / ||w_k||_2` estimates distance to the linearized boundary. Let's work through a complete example. Suppose `|f_k| = 5.3` (the deficit from above) and `||w_k||_2 = 2.0` (the gradient magnitude). Then `pert_k = 5.3 / 2.0 = 2.65`. This means moving 2.65 units in direction `w_k` would reach the linearized decision boundary between these two classes.

What does the gradient magnitude `||w_k||_2 = 2.0` tell us? It indicates how rapidly the score gap changes as we move in direction `w_k`. A larger magnitude (say 4.0) means the gap closes faster, requiring a shorter distance: `pert_k = 5.3 / 4.0 = 1.325` units. A smaller magnitude (say 1.0) means the gap closes slowly, requiring more distance: `pert_k = 5.3 / 1.0 = 5.3` units. The small constant `1e-10` prevents division by zero if the gradient happens to be zero, though this rarely occurs in practice with well-trained networks.

### Tracking the Closest Boundary

Code: python

```python
            if pert_k < pert:
                pert = pert_k
                w = w_k
```

After computing the distance ratio `pert_k` for this candidate class, we check if it's smaller than the current minimum `pert` (initialized to infinity before the loop). If yes, we update both `pert` to the new minimum distance and `w` to the corresponding gradient direction. By the end of the loop, `pert` holds the smallest distance among all candidates, and `w` points toward the closest decision boundary. This greedy selection ensures each iteration takes the minimal step needed to approach any boundary, not just a predetermined target class.

## Computing and Applying the Perturbation

Once the closest boundary is identified, the algorithm computes the minimal step to reach it and applies this perturbation with overshoot to ensure the true boundary is crossed.

Code: python

```python
        # Minimal step for the selected direction
        r_i = (pert + 1e-4) * w / (torch.norm(w.flatten()) + 1e-10)
        r_tot = r_tot + r_i

        # Apply with overshoot to ensure crossing
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1

    return r_tot, loop_i, label, k_i, pert_image
```

The perturbation `r_i` is computed by normalizing the gradient direction `w` to unit length and scaling it by the distance `pert`. The small constant `1e-4` added to `pert` prevents numerical issues when the distance is very small, while `1e-10` in the denominator prevents division by zero. The perturbation is accumulated in `r_tot`, tracking the total change from the original image.

The multiplication by `(1 + overshoot)` applies the overshoot factor discussed earlier, ensuring the algorithm reliably crosses the true non-linear boundary rather than merely approaching it. The function returns five values: the total perturbation `r_tot` representing the complete adversarial noise pattern, the number of iterations `loop_i` performed before success or timeout, the original class label, the final adversarial class label `k_i`, and the complete perturbed image ready for evaluation.

## Adaptive Behavior and Optimization

The algorithm adapts to the local geometry. In regions where the decision boundary is nearly linear, it might converge in just one or two iterations. Where the boundary curves significantly, it automatically takes more iterations, each adjusting the path to follow the boundary's contours. The accumulated perturbation `r_tot` represents the complete path from the original point to the adversarial example, approximating the minimal trajectory through high-dimensional space.

An important optimization consideration is the `num_classes` parameter. While we could theoretically consider all classes in the dataset, in practice, considering only the top few classes (based on the initial prediction scores) is often sufficient and significantly reduces computation time. The original paper suggests using 10 classes for datasets like ImageNet. For MNIST with only 10 total classes, we consider all of them, but for ImageNet with 1000 classes, limiting to the top 10 reduces gradient computations by 99% while maintaining attack effectiveness.

