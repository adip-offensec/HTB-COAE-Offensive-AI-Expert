
Each FISTA iteration orchestrates gradient descent on smooth terms with proximal operations for sparsity, accelerated by momentum. Binary search wraps this optimization, automatically tuning the adversarial pressure to find minimal perturbations. These nested loops create ElasticNet's adaptive behavior - adjusting both perturbation refinement (FISTA) and attack strength (binary search) to each example's difficulty.

## The Complete FISTA Iteration

To execute one FISTA step, we evaluate the loss at the momentum point (not the current solution), apply gradient descent, enforce sparsity through soft thresholding, then extrapolate the next momentum position. This look-ahead evaluation is what accelerates convergence beyond standard proximal gradient methods.

Code: python

```python
def fista_step(
    adv_images,
    y_momentum,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    learning_rate,
    confidence,
    iteration,
    targeted=False,
    clip_min=0.0,
    clip_max=1.0,
):
    """
    Perform one complete FISTA iteration.

    Combines gradient computation, shrinkage-thresholding,
    and momentum update into a single optimization step.

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        y_momentum (torch.Tensor): Momentum point for gradient evaluation
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants per example
        model (nn.Module): Target model
        beta (float): L1 weight parameter
        learning_rate (float): FISTA step size
        confidence (float): Margin for misclassification
        iteration (int): Current FISTA iteration number for momentum calculation
        targeted (bool): Whether this is a targeted attack
        clip_min (float): Minimum valid pixel value
        clip_max (float): Maximum valid pixel value

    Returns:
        tuple: (new_adv_images, new_y_momentum, loss_value, distances)
    """
    # Ensure y_momentum requires gradients for backprop
    y_momentum = y_momentum.detach().requires_grad_(True)

    # Compute loss at momentum point
    total_loss, adversarial_loss, distances = compute_total_loss(
        y_momentum,
        original_images,
        labels_onehot,
        const,
        model,
        beta,
        confidence,
        targeted,
    )

    # Compute gradient of total loss w.r.t. momentum point
    total_loss_summed = total_loss.sum()
    total_loss_summed.backward()
    grad = y_momentum.grad

    # Gradient step: move in negative gradient direction
    y_new = y_momentum - learning_rate * grad

    # Apply shrinkage-thresholding (proximal operator for L1)
    adv_new = apply_shrinkage_thresholding(
        y_new, original_images, learning_rate * beta, clip_min, clip_max
    )

    # Compute momentum coefficient
    momentum_coef = compute_fista_momentum(iteration)

    # Update momentum point for next iteration
    y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)

    return adv_new, y_new_momentum, total_loss_summed.item(), distances
```

Why detach and then immediately require gradients again? The preparation serves a critical purpose. The `detach()` call breaks any existing gradient tape from previous iterations, severing the computational graph that PyTorch built during the last FISTA step. Without this break, gradients would accumulate incorrectly across iterations, creating memory leaks and producing wrong gradient values. The `requires_grad_(True)` call then tells PyTorch to start fresh, tracking operations on this tensor for the current iteration only.

Where do we evaluate the loss? At the momentum point y(k), not the current solution x(k). This look-ahead evaluation is what makes FISTA faster than standard proximal gradient methods. By computing gradients at an extrapolated point that anticipates where optimization is heading, FISTA better exploits momentum in consistent directions. The gradient tells us which direction reduces loss from this forward-looking position.

Taking the gradient step, we move against the gradient to decrease loss: `y_new = y_momentum - learning_rate * grad`. The learning rate controls step size, balancing between fast convergence (large steps) and stability (small steps). Next, shrinkage handles the non-smooth L1 term. Small perturbations die (values below threshold β get zeroed). Larger perturbations survive with reduction (values above β get shrunk by exactly β).

Finally, momentum extrapolation anticipates where we're heading. The term `(adv_new - adv_images)` represents the change made in this iteration. Multiplying by `momentum_coef` and adding to `adv_new` produces a point slightly ahead of the current solution: `y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)`. The next iteration will compute gradients at this look-ahead point, building on the progress direction established by this step.

## Checking Attack Success

Binary search requires feedback about whether current perturbations successfully fool the model. We compare model predictions on adversarial examples to true labels through the success check, determining which examples achieved misclassification. This boolean feedback guides the search toward minimal perturbations sufficient for attack success.

Code: python

```python
def check_attack_success(adv_images, labels, model, targeted=False):
    """
    Verify whether adversarial examples achieve misclassification.

    Compares model predictions on adversarial images to target labels.

    Parameters:
        adv_images (torch.Tensor): Adversarial images to evaluate
        labels (torch.Tensor): True labels (or target labels if targeted)
        model (nn.Module): Target model
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Boolean mask indicating successful attacks (batch_size,)
    """
    with torch.no_grad():
        outputs = model(adv_images)
        predictions = outputs.argmax(dim=1)

        if targeted:
            # Success = prediction matches target
            success = predictions.eq(labels)
        else:
            # Success = prediction differs from true label
            success = predictions.ne(labels)

    return success
```

We run the success check in inference mode without gradient tracking (`torch.no_grad()`). No optimization occurs here; we simply evaluate whether the current perturbations achieve our goal. The forward pass produces logits, which we convert to class predictions via `argmax`.

For untargeted attacks, success means the prediction differs from the true label. Any misclassification counts as success; we don't care which wrong class the model predicts. For targeted attacks, success requires the prediction to match a specific target class. The targeted threat model is more constrained and typically requires larger perturbations.

This success criterion is binary: an example either fools the model or it doesn't. No notion of "partially successful" or "almost working" exists. This hard threshold contrasts with the soft margin in the adversarial loss. During optimization, the loss provides smooth gradients even when attacks haven't succeeded. After optimization, the success check provides definitive feedback for the binary search.

## Updating Binary Search Bounds

The binary search maintains lower and upper bounds on the trade-off constant c for each example. Successful attacks indicate c was sufficient (possibly too large), so we lower the upper bound. Failed attacks indicate c was insufficient, so we raise the lower bound. After several iterations, the bounds converge to the minimal c sufficient for attack success.

Code: python

```python
def update_binary_search_bounds(lower_bound, upper_bound, const, success_mask):
    """
    Update binary search bounds based on attack success.

    Successful attacks lower upper bound (c was sufficient).
    Failed attacks raise lower bound (c was insufficient).

    Parameters:
        lower_bound (torch.Tensor): Lower bounds on c per example
        upper_bound (torch.Tensor): Upper bounds on c per example
        const (torch.Tensor): Current c values per example
        success_mask (torch.Tensor): Boolean mask of successful attacks

    Returns:
        tuple: (new_lower_bound, new_upper_bound, new_const)
    """
    # Process each example individually
    for i in range(len(success_mask)):
        if success_mask[i]:
            # Success: try smaller c
            upper_bound[i] = min(upper_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
        else:
            # Failure: need larger c
            lower_bound[i] = max(lower_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                const[i] *= 10  # Exponential increase for persistent failures

    return lower_bound, upper_bound, const
```

The bisection mechanism from ElasticNet operates here through per-example bounds. Processing examples independently accommodates heterogeneous robustness: some examples lie far from decision boundaries and resist perturbation (needing large c values), while others sit near boundaries and flip easily (requiring small c values).

The implementation follows standard bisection logic. Successful attacks tighten the upper bound to the current constant, signaling we can try smaller values. Failed attacks raise the lower bound, indicating this constant was insufficient. When both bounds are finite, we bisect by computing their midpoint. This produces the logarithmic convergence described in ElasticNet: each iteration halves the search interval.

Exponential growth (line 176) handles the unbounded case where no success has occurred yet. Multiplying c by 10 each iteration explores the scale space efficiently (0.001,0.01,0.1,1,10,100,...) without wasting iterations on tiny increments. Once we find success at any scale, bisection takes over to refine the estimate. To see this in practice: starting with lower=0, upper=1010, c=0.001, a success sets upper=0.001 and bisects to c=0.0005. A subsequent failure sets lower=0.0005 and bisects to c=0.00075. After five such steps, the interval narrows by 25=32×.

## Putting FISTA and Binary Search Together

Complete attack execution nests FISTA optimization inside binary search iterations. An outer loop adjusts the trade-off constants based on success feedback. An inner loop optimizes perturbations for fixed constants, then we evaluate results. This separation isolates two subproblems: minimizing distortion (FISTA) and tuning adversarial pressure (binary search).

Each binary search iteration runs FISTA to convergence (or a maximum iteration limit). The resulting adversarial examples undergo success checks, providing feedback for the binary search update. The updated constants seed the next FISTA optimization round, which starts fresh from the original images. This fresh start prevents perturbations from growing unboundedly across binary search iterations.

This interplay between loops creates adaptive behavior. Easy examples converge quickly: FISTA succeeds early, binary search finds small optimal constants after few iterations. Hard examples take longer: FISTA needs many iterations to achieve misclassification, binary search explores larger constants across many steps. The algorithm adapts automatically to each example's difficulty.

This adaptive complexity is unique to ElasticNet among the attacks you’ve studied. FGSM has no adaptation; it uses fixed ϵ for all examples. DeepFool adapts through iteration count but uses the same algorithm for all examples. ElasticNet adapts both through FISTA iterations (finding perturbations) and binary search iterations (tuning pressure), providing fine-grained control over the attack process.

The next section shows these components operating together in the complete attack execution. Binary search iterations wrap FISTA optimization, adjusting constants based on success feedback. Configuration sets hyperparameters. Sample selection identifies attack targets. Result analysis quantifies effectiveness. The nested loop structure becomes explicit, showing exactly when binary search updates occur, when FISTA restarts from originals, and how the best perturbations get tracked across iterations.

