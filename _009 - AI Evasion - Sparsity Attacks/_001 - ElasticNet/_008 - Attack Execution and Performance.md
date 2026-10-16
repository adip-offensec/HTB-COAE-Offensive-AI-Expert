
# Attack Execution and Performance

All the pieces are now in place: momentum computation, shrinkage thresholding, loss functions, FISTA iterations, success checks, and binary search updates. This section brings them together into the complete attack execution, showing how these functions compose to generate adversarial examples.

## Attack Configuration

The attack requires several hyperparameters controlling different aspects of the optimization. The `beta` parameter balances L1 and L2 regularization, controlling sparsity. The learning rate determines FISTA step size. The iteration counts control optimization duration. The initial constant seeds the binary search. Together, these parameters define the attack strategy.

Code: python

```python
# Attack hyperparameters
config = {
    "beta": 0.01,  # L1 vs L2 trade-off (higher = sparser)
    "confidence": 0,  # Margin for misclassification
    "learning_rate": 0.01,  # FISTA step size
    "max_iterations": 1000,  # FISTA iterations per binary search
    "binary_search_steps": 5,  # Number of binary search iterations
    "initial_const": 0.001,  # Starting trade-off constant
    "clip_min": 0.0,  # Minimum pixel value
    "clip_max": 1.0,  # Maximum pixel value
}

print("\nAttack Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
```

How sparse should perturbations be? Setting `beta = 0.01` provides modest sparsity bias without sacrificing too much attack effectiveness. Larger values like 0.05-0.1 would create dramatically sparser perturbations by aggressively penalizing the L1 norm, forcing FISTA to zero out more pixels. Smaller values like 0.001-0.005 would produce nearly dense perturbations, behaving almost like pure L2 attacks where every pixel gets modified slightly. This middle-ground value balances sparsity with attack effectiveness.

Zero confidence means minimal misclassification criteria. With `confidence = 0`, any competitor class exceeding the true class by even 0.001 counts as success. Positive values would demand more confident misclassification: `confidence = 5` requires the competitor to exceed the true class by at least 5 logit units, producing more robust adversarial examples at the cost of larger perturbations. We use zero to minimize the perturbation budget needed for attack success.

Convergence speed versus stability depends critically on learning rate. Our choice of 0.01 strikes a balance. Push it higher to 0.05-0.1 and you risk overshooting, where perturbations oscillate wildly rather than converging smoothly to a solution. Drop it to 0.001-0.005 and convergence becomes painfully slow, requiring 2000-5000 iterations instead of 1000. For MNIST-scale problems, 0.01 typically works well without requiring problem-specific tuning.

## Sample Selection and Initialization

We select correctly classified examples from the test set for attack demonstrations. The attack targets only examples the model already classifies correctly - attacking misclassified examples makes no sense. This selection ensures we're testing the model's robustness on inputs it handles successfully under normal conditions.

Code: python

```python
print("\nSelecting correctly classified samples for attack...")
model.eval()
num_samples = 20

for data, targets in test_loader:
    data, targets = data.to(device), targets.to(device)
    outputs = model(data)
    predictions = outputs.argmax(dim=1)

    # Select correctly classified samples
    correct_mask = predictions.eq(targets)
    attack_data = data[correct_mask][:num_samples]
    attack_targets = targets[correct_mask][:num_samples]

    if len(attack_data) >= num_samples:
        print(f"Selected {len(attack_data)} samples")
        break
```

We evaluate the model on batches from the test loader, accumulating correctly classified examples until we have enough. How does the filtering work? `predictions.eq(targets)` produces a boolean tensor marking which examples match their true labels, with `True` for correct predictions and `False` for errors. Indexing `data[correct_mask]` with this boolean tensor extracts only the correctly classified examples, discarding misclassified ones automatically through PyTorch's boolean indexing mechanism.

Before running the nested optimization loops, we initialize variables tracking the attack state. These include copies of original images, one-hot encoded labels, binary search bounds, and storage for best perturbations found so far:

Code: python

```python
print("\nPerforming ElasticNet attack...")
print("This may take several minutes due to iterative optimization...\n")

batch_size = len(attack_data)
original_images = attack_data.clone()

# Convert labels to one-hot encoding
labels_onehot = torch.zeros(batch_size, 10).to(device)
labels_onehot.scatter_(1, attack_targets.unsqueeze(1), 1)

# Initialize binary search bounds
lower_bound = torch.zeros(batch_size).to(device)
upper_bound = torch.ones(batch_size).to(device) * 1e10
const = torch.ones(batch_size).to(device) * config["initial_const"]

# Track best adversarial examples
best_adv = original_images.clone()
best_l2 = torch.ones(batch_size).to(device) * 1e10
```

Converting integer labels to one-hot encoding creates binary vectors with 1 at the true class index and 0 elsewhere. `labels_onehot.scatter_(1, attack_targets.unsqueeze(1), 1)` places 1s at the appropriate positions by scattering along dimension 1 (the class dimension). Why use one-hot instead of integers? The adversarial loss computation needs `labels_onehot` as a mask to extract true class logits efficiently, avoiding expensive indexing operations in the inner optimization loop.

Binary search bounds need wide initial intervals. Lower bounds start at zero (no adversarial pressure), upper bounds at 1e10 (essentially infinite pressure). What happens if we start with narrow intervals? The search might converge to suboptimal constants that either fail to produce adversarial examples or produce unnecessarily large perturbations. Starting wide ensures we cover the full range of constants the attack might need. The current constants initialize small at 0.001, which suffices for easy examples and gets increased automatically through binary search for harder ones.

## The Binary Search and FISTA Loops

Now we execute the nested optimization. The outer binary search loop iterates over trade-off constant refinements. Each binary search iteration runs an inner FISTA optimization loop to completion, evaluates success, and updates constants:

Code: python

```python
# Binary search over trade-off constant c
for binary_step in range(config["binary_search_steps"]):
    print(f"Binary search step {binary_step + 1}/{config['binary_search_steps']}")

    # Initialize FISTA from original images
    adv_images = original_images.clone().detach()
    y_momentum = adv_images.clone()

    # FISTA optimization loop
    for iteration in range(config["max_iterations"]):
        # Perform one FISTA step
        adv_images, y_momentum, loss, distances = fista_step(
            adv_images,
            y_momentum,
            original_images,
            labels_onehot,
            const,
            model,
            config["beta"],
            config["learning_rate"],
            config["confidence"],
            iteration,
            targeted=False,
            clip_min=config["clip_min"],
            clip_max=config["clip_max"],
        )

    # Check which examples successfully fooled the model
    success_mask = check_attack_success(
        adv_images, attack_targets, model, targeted=False
    )

    # Update best adversarial examples
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, config["beta"]
    )

    for i in range(batch_size):
        if success_mask[i] and l2_dist[i] < best_l2[i]:
            best_adv[i] = adv_images[i]
            best_l2[i] = l2_dist[i]

    # Update binary search bounds
    lower_bound, upper_bound, const = update_binary_search_bounds(
        lower_bound, upper_bound, const, success_mask
    )

    # Progress reporting
    num_success = success_mask.sum().item()
    print(f"  Successfully generated {num_success}/{batch_size} adversarial examples\n")
```

Why reinitialize from original images at each binary search iteration? This fresh start prevents perturbations from accumulating across iterations with different constants. Without reinitialization, perturbations optimized for c=0.001 would persist when testing c=0.01, contaminating the optimization and producing unreliable results. Starting fresh ensures each constant gets a fair evaluation.

FISTA runs up to 1000 iterations, with each calling `fista_step` to perform gradient computation, shrinkage thresholding, and momentum updates. Once optimization completes, we evaluate success and update the binary search. Which examples achieved misclassification? `check_attack_success` compares model predictions against target labels. The best perturbation update processes each example independently, comparing current L2 distortion to previous bests. Only successful attacks with smaller distortion than what we’ve seen before get saved, ensuring we track the minimal perturbation for each example.

Adjusting bounds and constants happens through `update_binary_search_bounds`. Successful examples lower their upper bounds (we found a working constant, so we can try smaller values) and bisect the interval to search the lower half. Failed examples raise their lower bounds (this constant was too small), using exponential growth if no success has occurred yet or bisection if the interval is already bounded from above.

## Computing and Analyzing Results

After binary search completes, we evaluate the final adversarial examples and compute comprehensive statistics:

Code: python

```python
# Evaluate final adversarial examples
with torch.no_grad():
    adv_outputs = model(best_adv)
    adv_predictions = adv_outputs.argmax(dim=1)

    # Calculate success rate
    final_success = adv_predictions.ne(attack_targets)
    success_rate = final_success.float().mean().item() * 100

    # Calculate final distortions
    l1_dist, l2_dist, elastic_dist = compute_distances(
        best_adv, original_images, config["beta"]
    )
    linf_dist = torch.max(
        torch.abs(best_adv - original_images).view(batch_size, -1), dim=1
    )[0]

# Display results
print("=" * 60)
print("ElasticNet Attack Results:")
print("=" * 60)
print(f"Success Rate: {success_rate:.2f}% ({final_success.sum()}/{batch_size})")
print(f"Average L1 Distortion: {l1_dist.mean().item():.4f}")
print(f"Average Squared L2 Distortion: {l2_dist.mean().item():.4f}")
print(f"Average L∞ Distortion: {linf_dist.mean().item():.4f}")
print(f"Average Elastic Distortion: {elastic_dist.mean().item():.4f}")
print("=" * 60)
```

Attack effectiveness emerges across multiple metrics. Success rate quantifies what percentage of examples were successfully misclassified, providing an overall measure of attack reliability. Distortion metrics reveal perturbation magnitude from different geometric perspectives: L1 counts total pixel changes, L2 measures Euclidean distance, L∞ captures maximum single-pixel deviation, and elastic distance combines L1 and L2 through the β weighting. Together, these reveal both the size and structural characteristics of our modifications.

## Performance

The complete attack execution reveals ElasticNet's computational profile. For 20 MNIST examples with 5 binary search steps and 1000 FISTA iterations per step, the attack performs roughly 5,000 forward and 5,000 backward steps for the batch (about 100,000 per-image evaluations at batch size 20). On a modern GPU (RTX 3090 or better), this typically takes 2-3 minutes; on CPU, 30-60 minutes. These figures are representative for MNIST-scale CNNs and vary with hardware and implementation details.

Binary search convergence directly impacts total runtime. Reducing binary search steps from 9 to 5 cuts computation by 45% but increases final distortion by 5-15% because constant tuning is less precise. Increasing steps to 9 reduces distortion by 3-8% but costs 80% more computation. The sweet spot depends on use case: adversarial training benefits from speed (3-5 steps), while robustness evaluation demands precision (7-9 steps).

Parameter sensitivity affects attack reliability across different models and datasets. The learning rate shows highest sensitivity: values too large (>0.05 for MNIST) cause divergence where perturbations oscillate rather than converge. Values too small (<0.001) require 2000-5000 iterations for convergence, quadrupling runtime. The beta parameter shows moderate sensitivity: doubling from 0.01 to 0.02 increases sparsity by 10-15 percentage points but may reduce success rate by 5-10% on hard examples.

Batch size trades memory for efficiency. Processing 20 examples simultaneously uses 400 MB GPU memory but amortizes optimization overhead. Increasing to 50 examples improves throughput by 15-25% but requires 1 GB memory. Decreasing to 10 examples halves memory usage to 200 MB but reduces throughput by 10-15% due to underutilized GPU cores.

The next section develops visualizations that expose these patterns visually. Plotting original versus adversarial images reveals where perturbations concentrate. Distortion distributions show variation across examples. Success rate versus distortion curves characterize the effectiveness-imperceptibility tradeoff. Sparsity heatmaps expose which pixels ElasticNet modifies. These visualizations complement the quantitative metrics, providing qualitative insights about how the attack operates.

