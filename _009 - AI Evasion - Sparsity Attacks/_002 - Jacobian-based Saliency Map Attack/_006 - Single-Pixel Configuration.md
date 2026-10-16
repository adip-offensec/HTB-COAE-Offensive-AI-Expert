
# Single-Pixel Configuration

The attack's behavior depends critically on parameter choices. Understanding trade-offs helps configure attacks effectively for different scenarios. This section explores how step size and feature budget parameters affect attack success, efficiency, and perturbation character.

## Step Size Configuration

The step size controls how much we modify each selected pixel per iteration. Testing across multiple values reveals the trade-off between attack speed and perturbation subtlety. Let's experiment systematically on the first collected sample:

Code: python

```python
theta_values = [0.10, 0.25, 0.50, 1.00]

print("\nStep Size Analysis:")
print(f"{'theta':<8} {'Iterations':<12} {'Pixels':<8} {'Result':<8}")
print("="*42)
```

Output:

Code: txt

```txt
Step Size Analysis:
theta    Iterations   Pixels   Result
==========================================
```

We'll test four step sizes ranging from subtle (0.10) to aggressive (1.00), analyzing how each affects iteration count and success. The formatted table header establishes columns for the parameter value, iteration count required, total pixels modified, and final attack outcome.

### Running Step Size Experiments

For each theta value, we execute a complete attack and record the results:

Code: python

```python
for theta_test in theta_values:
    # Initialize fresh attack
    x_test = original_images[0].clone().detach()
    search_space_test = initialize_search_space(x_test.shape)
    config_test = {**config, 'theta': theta_test}
    pixels_mod = 0

    # Run attack loop
    for iteration in range(100):
        if check_target_reached(x_test, target_labels[0], model):
            break
        if pixels_mod >= int(config_test['gamma'] * 784):
            break

        jacobian = compute_jacobian_matrix(x_test, model, 10, config_test['wrt'])
        alpha = extract_target_gradient(jacobian, target_labels[0])
        beta = extract_other_gradients(jacobian, target_labels[0])
        alpha_masked = apply_search_mask(alpha, search_space_test)
        beta_masked = apply_search_mask(beta, search_space_test)

        inc_scores = score_increase_saliency(alpha_masked, beta_masked)
        dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
        pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

        if saliency <= 0:
            break

        x_test = apply_single_pixel_perturbation(
            x_test, pixel_idx, config_test['theta'], increase, 0.0, 1.0
        )

        search_space_test[pixel_idx] = False
        search_space_test = remove_saturated_pixels(search_space_test, x_test, 0.0, 1.0)
        pixels_mod += 1
```

Each experiment starts fresh by cloning the original image and initializing a new search space, ensuring results don't carry over between tests. The dictionary unpacking `{**config, 'theta': theta_test}` creates a new configuration copying all base settings while overriding only the theta parameter. This preserves `gamma`, `max_iter`, and other parameters across experiments. The attack loop follows the same structure used throughout this section: check termination, compute gradients, score pixels, select best, apply perturbation, update state. Masking the modified pixel (`search_space_test[pixel_idx] = False`) prevents repeat selection even before saturation, which is critical for these experiments since small theta values take multiple modifications to saturate pixels.

### Recording and Displaying Results

After each attack completes, we check success and display the outcome:

Code: python

```python
    success = check_target_reached(x_test, target_labels[0], model)
    result = "SUCCESS" if success else "FAILED"
    print(f"{theta_test:<8.2f} {iteration+1:<12} {pixels_mod:<8} {result:<8}")

print("="*42)
```

Output:

Code: txt

```txt
0.10     100          100      FAILED
0.25     100          100      FAILED
0.50     100          100      FAILED
1.00     64           63       SUCCESS
=========================================
```

String formatting with `.2f` displays theta values to two decimal places, maintaining consistent column widths. Adding one to the iteration variable accounts for zero-based indexing (if the loop exits with `iteration=63`, we completed 64 iterations). The result string provides immediate visual confirmation of attack effectiveness.

### Interpreting Step Size Trade-offs

The results reveal a critical threshold effect: small theta values (0.10, 0.25, 0.50) all fail within 100 iterations, while `theta=1.0` succeeds in just 64 iterations. With `theta=1.0`, each selected pixel saturates immediately (jumping from 0.0 to 1.0 or 1.0 to 0.0), enabling the attack to accumulate enough perturbation magnitude to cross the decision boundary. In contrast, smaller theta values modify pixels incrementally, requiring many iterations to build up sufficient change.

Notice that pixel counts match iteration counts almost exactly (63 pixels for 64 iterations with `theta=1.0`). This one-to-one correspondence occurs because we mask each pixel after modification, preventing reselection. The attack selects a new pixel at each iteration rather than repeatedly modifying high-saliency pixels.

This sample demonstrates why `theta=1.0` is recommended for MNIST when attack success is prioritized over perturbation subtlety. The aggressive per-pixel modifications create visible artifacts but enable misclassification with fewer total pixel changes. Smaller theta values create subtler perturbations but often fail entirely, as shown here.

## Feature Budget Configuration

The `gamma` parameter controls sparsity by limiting the fraction of pixels we can modify:

Code: python

```python
gamma_values = [0.10, 0.15, 0.20, 0.30]

print("\nFeature Budget Analysis:")
print(f"{'gamma':<8} {'Max Pixels':<12} {'Percentage':<12}")
print("="*38)

for gamma_test in gamma_values:
    max_pixels = int(gamma_test * 784)
    print(f"{gamma_test:<8.2f} {max_pixels:<12} {gamma_test*100:<12.0f}%")

print("="*38)
```

Output:

Code: txt

```txt
Feature Budget Analysis:
gamma    Max Pixels   Percentage
======================================
0.10     78           10%
0.15     117          15%
0.20     157          20%
0.30     235          30%
======================================
```

In practice, successful attacks typically use a portion of the available budget. Our attacks modified 73.6 pixels on average under a `gamma=0.15` budget of 117 pixels, consuming about 63% of the available budget. Successful attacks ranged from 33 to 92 pixels, while the three failed attacks exhausted the full 100 iterations without reaching their targets.

Tighter budgets (`gamma=0.10`, allowing 78 pixels) would constrain some successful attacks that required 86-92 pixels, potentially reducing the success rate. Looser budgets (`gamma=0.30`, allowing 235 pixels) would provide additional headroom but may not improve success rates since most successful attacks already complete well below the current 117-pixel limit. The value `gamma=0.15` provides a reasonable balance, allowing sufficient modifications for most attacks while maintaining sparsity.

## Reflecting on Configuration Choices

Across 10 samples, single-pixel JSMA with `theta=0.25` achieved 70% success (7/10 samples), modifying a mean of 73.6 pixels. Three attacks failed after exhausting the 100-iteration budget without crossing decision boundaries. The theta experiment revealed that `theta=1.0` succeeds where smaller values fail, achieving success in 64 iterations where `theta=0.25` failed after 100 iterations on the same sample. Saliency scores declined steadily during attacks, suggesting diminishing returns from sequential pixel selection.

While single-pixel JSMA demonstrates moderate effectiveness with `theta=0.25`, the canonical JSMA algorithm uses pairwise feature selection to improve both success rates and efficiency, implemented in the following sections. Pairwise selection captures feature interactions by modifying two pixels simultaneously, often finding more effective perturbation combinations than single-pixel greedy selection.

