
# Single-Pixel Batch Evaluation

Single-Pixel Attack demonstrated single-pixel JSMA mechanics on one sample, which failed despite modifying 100 pixels and reaching minimal target confidence. A single example can't tell us whether this failure typifies single-pixel JSMA or represents an unusually difficult sample. To characterize typical performance and identify patterns in success rates, sparsity, and convergence behavior, we now evaluate across multiple samples.

This section shows how to run batch attacks, compute summary statistics, and measure sparsity across a representative test set.

## Batch Attack Execution

Running the attack on multiple samples reveals typical performance characteristics and variability across the dataset. We need to collect correctly classified samples, attack each one systematically, and analyze the aggregate results to understand single-pixel JSMA's effectiveness.

### Collecting Test Samples

To evaluate attack performance across diverse inputs, we'll collect 10 correctly classified samples from the test set. For each sample, we'll assign a target class offset by 5 positions to ensure challenging misclassification targets:

Code: python

```python
print("Collecting samples...")
samples_found = 0
target_count = 10

original_images = []
original_labels = []
target_labels = []
```

Setting up storage structures prepares us to accumulate samples as we encounter them. The `samples_found` counter tracks progress toward our target count, while the three lists maintain parallel arrays where index i corresponds to the same sample across all three (image at `original_images[i]` has label `original_labels[i]` and target `target_labels[i]`). This parallel array pattern simplifies iteration during the attack phase since we can loop over indices and access all related data with the same index.

### Scanning the Dataset

We iterate through test batches, filtering for correctly classified samples until we reach our target count:

Code: python

```python
for x_batch, y_batch in test_loader:
    if samples_found >= target_count:
        break

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        preds = model(x_batch).argmax(dim=1)
```

Checking the target count at the batch level enables early termination before processing unnecessary data. Moving batches to the device (GPU or CPU) ensures tensor operations run on the correct hardware. Wrapping prediction in `torch.no_grad()` disables gradient tracking since we're only evaluating, not training, saving memory and computation. The `argmax(dim=1)` operation finds the predicted class for each image in the batch by selecting the highest logit along the class dimension.

### Filtering and Storing Samples

Within each batch, we examine individual samples, keeping only those correctly classified:

Code: python

```python
    for i in range(x_batch.size(0)):
        if samples_found >= target_count:
            break

        if preds[i].item() != y_batch[i].item():
            continue

        x = x_batch[i:i+1]
        original_class = int(y_batch[i].item())
        target_class = (original_class + 5) % 10

        original_images.append(x)
        original_labels.append(original_class)
        target_labels.append(target_class)

        samples_found += 1
        print(f"  Sample {samples_found}: digit {original_class} → target {target_class}")

print(f"\nCollected {len(original_images)} samples")
```

Output:

Code: txt

```txt
Collecting samples...
  Sample 1: digit 7 → target 2
  Sample 2: digit 2 → target 7
  Sample 3: digit 1 → target 6
  Sample 4: digit 0 → target 5
  Sample 5: digit 4 → target 9
  Sample 6: digit 1 → target 6
  Sample 7: digit 4 → target 9
  Sample 8: digit 9 → target 4
  Sample 9: digit 5 → target 0
  Sample 10: digit 9 → target 4

Collected 10 samples
```

The inner loop processes each sample in the current batch. Redundant target count checking ensures we stop immediately upon reaching our goal, even mid-batch. Comparing prediction to true label identifies correctly classified samples (the only ones suitable for targeted attacks, since already-misclassified samples don’t provide meaningful baselines). Slicing `x_batch[i:i+1]` preserves the batch dimension, yielding shape (1,1,28,28) rather than (1,28,28), which JSMA expects. The modular arithmetic `(original_class + 5) % 10` wraps around at 10, ensuring targets stay in range while providing semantic distance (digit 7 targets digit 2, challenging the model to cross the decision boundary between dissimilar classes).

### Initializing Attack Results Storage

With our test set ready, we prepare to execute attacks and track comprehensive results:

Code: python

```python
print("\nRunning attacks...")
print(f"{'#':<4} {'Orig→Tgt':<10} {'Result':<10} {'Pixels':<8} {'Iters':<8}")
print("="*46)

results = {
    'adversarial': [],
    'success': [],
    'pixels_modified': [],
    'iterations': []
}
```

Output:

Code: txt

```txt
Running attacks...
#    Orig→Tgt   Result     Pixels   Iters
==============================================
```

The results dictionary uses lists to accumulate per-sample metrics, maintaining the same parallel array pattern as the input data. Storing adversarial examples enables post-attack visualization and perturbation analysis. Success flags support aggregate success rate calculation. Pixel counts reveal sparsity characteristics. Iteration counts expose convergence behavior and computational cost. The formatted header establishes columnar output for tracking progress across all 10 attacks.

### Running Attacks on All Samples

We execute the complete single-pixel JSMA attack on each collected sample, using the exact loop structure from the previous section:

Code: python

```python
for idx in range(len(original_images)):
    x = original_images[idx]
    orig_class = original_labels[idx]
    tgt_class = target_labels[idx]

    # Initialize attack state
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    pixels_mod = 0

    # Attack loop
    for iteration in range(config['max_iter']):
        if check_target_reached(x_adv, tgt_class, model):
            break
        if pixels_mod >= int(config['gamma'] * 784):
            break

        jacobian = compute_jacobian_matrix(x_adv, model, 10, config['wrt'])
        alpha = extract_target_gradient(jacobian, tgt_class)
        beta = extract_other_gradients(jacobian, tgt_class)
        alpha_masked = apply_search_mask(alpha, search_space)
        beta_masked = apply_search_mask(beta, search_space)

        inc_scores = score_increase_saliency(alpha_masked, beta_masked)
        dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
        pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

        if saliency <= 0:
            break

        x_adv = apply_single_pixel_perturbation(
            x_adv, pixel_idx, config['theta'], increase,
            config['clip_min'], config['clip_max']
        )

        search_space = remove_saturated_pixels(search_space, x_adv, 0.0, 1.0)
        pixels_mod += 1
```

The outer loop iterates over sample indices, extracting the corresponding image, true label, and target from the parallel arrays. Each attack starts fresh with a cloned adversarial image (ensuring we don't contaminate across samples), initialized search space (all pixels available), and zero pixel counter. The inner attack loop replicates the structure from the previous section: check termination conditions, compute Jacobian, extract and mask gradients, score both directions, select best pixel, apply perturbation, update search space. Notice we removed individual pixel masking (`search_space[pixel_idx] = False`) from the previous version; here we rely solely on saturation detection via `remove_saturated_pixels`, which automatically handles boundary cases while still preventing repeat selections through the modification tracking inherent in the pixel values themselves.

### Storing Results and Displaying Progress

After each attack completes, we record all metrics and display a summary row:

Code: python

```python
    # Record results
    success = check_target_reached(x_adv, tgt_class, model)
    pred = model(x_adv).argmax(dim=1).item()

    results['adversarial'].append(x_adv)
    results['success'].append(success)
    results['pixels_modified'].append(pixels_mod)
    results['iterations'].append(iteration + 1)

    # Display progress
    status = "✓" if success else "✗"
    print(f"{idx+1:<4} {orig_class}→{tgt_class:<8} {status:<10} {pixels_mod:<8} {iteration+1:<8}")

print("="*46)
```

Output:

Code: txt

```txt
1    7→2        ✗          100      100
2    2→7        ✗          100      100
3    1→6        ✓          67       68
4    0→5        ✓          33       34
5    4→9        ✓          58       59
6    1→6        ✓          92       93
7    4→9        ✓          60       61
8    9→4        ✓          86       87
9    5→0        ✗          100      100
10   9→4        ✓          40       41
==============================================
```

The final success check performs one last forward pass to definitively determine whether the adversarial example forces misclassification to the target. This redundant check (we already checked in the loop) provides consistent results independent of termination reason. Extracting the final prediction enables debugging when attacks fail (did we stay at the original class, drift to an unintended class, or nearly reach the target?). Adding `iteration + 1` to the results corrects for zero-based indexing (if the loop variable ends at 99, we ran 100 iterations). The status symbol provides immediate visual feedback: checkmarks for success, X marks for failure. Column alignment with fixed field widths keeps the table readable across varying number widths.

Seven out of 10 samples succeeded with the current configuration (`theta=0.25`, `max_iter=100`). Samples 1, 2, and 9 failed after exhausting the 100 iteration limit. Successful attacks required between 33 and 92 pixels, demonstrating that single-pixel JSMA can achieve misclassification when given sufficient iterations and appropriate samples, though success varies significantly across the dataset.

## Attack Summary Statistics

Let's compute aggregate statistics to characterize overall attack performance:

Code: python

```python
success_count = sum(results['success'])
total_samples = len(results['success'])
success_rate = 100.0 * success_count / total_samples

pixels = results['pixels_modified']
mean_pixels = np.mean(pixels)
median_pixels = np.median(pixels)
std_pixels = np.std(pixels)
min_pixels = np.min(pixels)
max_pixels = np.max(pixels)

sparsity_pct = 100.0 * mean_pixels / 784

print("\nAttack Summary:")
print("="*50)
print(f"Success rate:        {success_count}/{total_samples} ({success_rate:.1f}%)")
print(f"\nPixels Modified:")
print(f"  Mean:   {mean_pixels:.1f} ± {std_pixels:.1f}")
print(f"  Median: {median_pixels:.1f}")
print(f"  Range:  [{min_pixels}, {max_pixels}]")
print(f"\nSparsity: {mean_pixels:.1f} / 784 = {sparsity_pct:.2f}%")
print("="*50)
```

Output:

Code: txt

```txt
Attack Summary:
=================================================
Success rate:        7/10 (70.0%)

Pixels Modified:
  Mean:   73.6 ± 24.2
  Median: 76.5
  Range:  [33, 100]

Sparsity: 73.6 / 784 = 9.39%
=================================================
```

The results show moderate success: 70% success rate across the 10 samples, with 7 successful attacks and 3 failures. The attacks modified an average of 73.6 pixels (9.39% of the image). The median of 76.5 indicates typical attacks fall in the mid-range of pixel modifications. Successful attacks ranged from 33 to 92 pixels, demonstrating variable efficiency. The three failed samples (1, 2, and 9) all exhausted the 100-iteration limit, suggesting they require either more iterations, larger step sizes, or represent particularly challenging misclassification targets where single-pixel selection struggles to find effective perturbation combinations.

## Perturbation Analysis

Beyond success rates, we need to understand the perturbation character: how sparse are they really, and how much do individual pixels change? Let's examine all four distance metrics for the first sample to reveal the trade-offs inherent in L0-minimizing attacks:

Code: python

```python
x_orig = original_images[0].cpu().numpy()
x_adv = results['adversarial'][0].cpu().numpy()

perturbation = x_adv - x_orig
pert_magnitude = np.abs(perturbation)

# Compute norms
l0_norm = np.count_nonzero(perturbation)
l1_norm = np.sum(pert_magnitude)
l2_norm = np.linalg.norm(perturbation)
linf_norm = np.max(pert_magnitude)

print(f"\nPerturbation Analysis (Sample 1):")
print(f"="*50)
print(f"L0 (pixels changed):     {l0_norm}")
print(f"L1 (sum of changes):     {l1_norm:.4f}")
print(f"L2 (euclidean distance): {l2_norm:.4f}")
print(f"L∞ (max change):         {linf_norm:.4f}")
print(f"="*50)
```

Output:

Code: txt

```txt
Perturbation Analysis (Sample 1):
=================================================
L0 (pixels changed):     51
L1 (sum of changes):     19.4422
L2 (euclidean distance): 3.5413
L∞ (max change):         0.9961
=================================================
```

Reading these norms reveals the perturbation character. L0=51 confirms exactly 51 pixels changed, representing 6.5% of the image. L2=3.5413 measures total perturbation magnitude, moderate for a 784-pixel image. L∞=0.9961 approaches the maximum possible change of 1.0, indicating at least one pixel saturated near its boundary. The L1 norm of 19.4422 divided by 51 pixels gives an average absolute change of 0.381 per modified pixel, suggesting many pixels received multiple modifications or large single modifications during the attack iterations.

The next section provides configuration guidelines, showing how different parameter choices (step size and feature budget) affect attack behavior and success rates.

