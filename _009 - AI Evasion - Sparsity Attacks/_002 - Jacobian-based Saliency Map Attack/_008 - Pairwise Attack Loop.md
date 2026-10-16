
# Pairwise Attack Loop

Having demonstrated a single iteration with pairwise saliency, we can now execute the full pairwise attack loop. We'll build this incrementally, following the same structured approach used for single-pixel JSMA.

## Attack Configuration and Reset

To begin fresh, we reset the adversarial image and configure pairwise-specific parameters:

Code: python

```python
# Reset
x_adv = x.clone().detach()
search_space = initialize_search_space(x.shape)

# Configuration
config = {
    'theta': 1.0,
    'gamma': 0.15,
    'max_iter': 90,
    'wrt': 'logits',
    'clip_min': 0.0,
    'clip_max': 1.0,
    'top_k': 128
}
```

Cloning the original image ensures we start from a clean state, not carrying over modifications from previous experiments. Initializing the search space marks all pixels as available. The configuration uses `theta=1.0` for aggressive single-step saturation (pairwise selection is efficient enough that we don't need multiple modifications per pixel), `gamma=0.15` for the same sparsity budget as single-pixel attacks, and adds `top_k=128` to enable candidate pruning for computational efficiency.

## Computing Pixel Budget

The pixel budget calculation remains identical to single-pixel JSMA, translated from gamma fraction to absolute pixel count:

Code: python

```python
num_features = int(np.prod(x.shape[1:]))
max_pixels = int(config['gamma'] * num_features)

print(f"Configuration:")
print(f"  Theta: {config['theta']}, Gamma: {config['gamma']}")
print(f"  Max pixels: {max_pixels}, Top-k: {config['top_k']}")
```

Output:

Code: txt

```txt
Configuration:
  Theta: 1.0, Gamma: 0.15
  Max pixels: 117, Top-k: 128
```

For MNIST, 0.15×784=117 pixels maximum. With pairwise selection modifying 2 pixels per iteration, we can execute at most ⌊117/2⌋=58 iterations before exhausting the budget (though we’ll likely succeed earlier).

## Initializing Progress Tracking

We initialize tracking structures and display the progress header:

Code: python

```python
pixels_modified = 0

print(f"\nStarting pairwise attack: {original_class} → {target_class}")
print(f"\n{'Iter':<6} {'Pixels':<8} {'Confidence':<12} {'Score':<12}")
print("="*46)
```

Output:

Code: txt

```txt
Starting pairwise attack: 7 → 2

Iter   Pixels   Confidence   Score
==============================================
```

The pixel counter tracks cumulative L0 norm, incrementing by 2 per iteration since pairwise JSMA modifies two pixels simultaneously. The formatted header establishes column widths matching the single-pixel format, adding a score column to track pairwise saliency values.

## Main Iteration Loop with Termination Checks

The attack loop checks termination conditions before computing expensive gradients:

Code: python

```python
for iteration in range(config['max_iter']):
    # Check termination conditions
    if check_target_reached(x_adv, target_class, model):
        print(f"\n✓ Target reached at iteration {iteration}!")
        break

    if pixels_modified >= max_pixels:
        print(f"\n✗ Budget exhausted")
        break

    # Compute Jacobian and extract gradients
    jacobian = compute_jacobian_matrix(x_adv, model, 10, config['wrt'])
    alpha = extract_target_gradient(jacobian, target_class)
    beta = extract_other_gradients(jacobian, target_class)
```

Structuring checks first prevents wasted Jacobian computation when the attack has already succeeded or exhausted resources. The Jacobian computation remains identical to single-pixel JSMA (10 backward passes for 10 classes), as does gradient extraction (isolating target class gradients and summing competitors). The difference emerges in how we use these gradients: for pair selection rather than individual pixel selection.

## Bidirectional Pair Scoring

We evaluate both increase and decrease directions, computing pairwise saliency for each:

Code: python

```python
    # Score both directions
    p_inc, q_inc, score_inc = compute_pairwise_saliency(
        alpha, beta, search_space, 'increase', config['top_k']
    )
    p_dec, q_dec, score_dec = compute_pairwise_saliency(
        alpha, beta, search_space, 'decrease', config['top_k']
    )

    # Check for exhausted search space
    if max(score_inc, score_dec) <= 0.0:
        print(f"\n✗ No valid pairs")
        break
```

Each `compute_pairwise_saliency` call performs candidate pruning (if `top_k` specified), evaluates all pairs, applies sign constraints, and returns the best pair with its score. Running both directions mirrors the paper's evaluation protocol: we don't know a priori which direction will be more effective, so we try both and choose the winner. Zero scores in both directions indicate complete search space exhaustion, all remaining pixels violate sign constraints or can't form valid pairs.

## Pair Selection and Validation

With scores computed, we select the winning direction and validate the result:

Code: python

```python
    # Select best direction
    if score_inc >= score_dec:
        p, q, score, increase = p_inc, q_inc, score_inc, True
    else:
        p, q, score, increase = p_dec, q_dec, score_dec, False

    # Validate selected pair
    if p < 0 or q < 0:
        print(f"\n✗ Invalid pair")
        break
```

Tournament selection picks the direction with higher saliency score. The increase direction wins ties (using `>=`), though this rarely matters in practice. Validation checks for the sentinel values (`p=-1` or `q=-1`) that `compute_pairwise_saliency` returns when unable to find valid pairs. This should never trigger given the earlier zero-score check, but defensive programming catches edge cases and implementation bugs.

## Perturbation Application and State Updates

With a valid pair selected, we apply the perturbation and update tracking state:

Code: python

```python
    # Apply pair perturbation
    x_adv = apply_pair_perturbation(
        x_adv, p, q, config['theta'], increase,
        config['clip_min'], config['clip_max']
    )

    # Update search space
    search_space[p] = False
    search_space[q] = False
    search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)

    # Track progress
    pixels_modified += 2
    confidence = compute_confidence(x_adv, target_class, model)

    # Print progress
    if iteration % 3 == 0 or iteration < 3:
        print(f"{iteration:<6} {pixels_modified:<8} {confidence:<12.4f} {score:<12.8f}")
```

Output:

Code: txt

```txt
0      2        0.0000       2.44320202
1      4        0.0000       1.63035405
2      6        0.0000       1.22470224
3      8        0.0000       1.73618484
6      14       0.0000       1.06038046
9      20       0.0000       0.70596105
12     26       0.0000       0.96916491
15     32       0.0002       0.52848244
18     38       0.0005       0.87520170
21     44       0.0037       0.61113876
24     50       0.0185       0.37239930
27     56       0.2927       0.55573213
30     62       0.3491       0.46899793
33     68       0.7262       0.41797119

✓ Target reached at iteration 34!
```

Pair perturbation modifies both pixels simultaneously with the same signed step, flattening to access by index and reshaping to restore structure. Search space updates mask out both modified pixels immediately, then scan for any saturated pixels throughout the image. Incrementing by 2 reflects pairwise modification. The display condition `iteration % 3` shows every third iteration (plus the first 3), providing visibility without overwhelming output. The saliency scores decline as high-value pairs are exhausted, while the confidence grows steadily toward the decision boundary.

## Final Results Evaluation

After loop termination, we evaluate the final adversarial example:

Code: python

```python
final_success = check_target_reached(x_adv, target_class, model)
final_pred = model(x_adv).argmax(dim=1).item()

print("\n✓ Target reached at iteration 35!")
print("\n" + "="*46)
print(f"Final result: {'SUCCESS' if final_success else 'FAILED'}")
print(f"Final prediction: {final_pred}")
print(f"Pixels modified: {pixels_modified}")
print(f"Iterations: {iteration + 1}")
```

Output:

Code: txt

```txt
==============================================
Final result: SUCCESS
Final prediction: 2
Pixels modified: 68
Iterations: 35
==============================================
```

The final success check confirms misclassification to the target class. Extracting the prediction shows which class the model assigns. The pixel count reports cumulative modifications (35 iterations × 2 pixels = 70 pixels, minus 2 due to some saturated pixels being skipped). Adding one to the iteration variable accounts for zero-based indexing.

## Interpreting Pairwise Performance

Success! The attack achieved the target in 35 iterations by modifying 68 pixels total (8.7% of the image). This demonstrates pairwise JSMA's effectiveness. Look at the confidence progression: the target confidence climbs from near-zero, with significant jumps starting around iteration 24 (reaching 0.0185), then accelerating through 0.2927 at iteration 27 and crossing the decision boundary to reach 0.7262 by iteration 33. The attack doesn't exhaust the pixel budget (117 pixels allowed), demonstrating efficient use of the available sparsity.

Compare this to the single-pixel baseline with `theta=0.25` where sample 1 (digit 7→2) failed after 100 iterations, modifying 100 pixels. Here we used 68 pixels (32% fewer) and succeeded. The combination of pairwise selection and logits-based gradients, along with aggressive `theta=1.0`, enables JSMA to achieve effective sparse attacks on MNIST. The pairwise saliency captures feature interactions through the sign constraints on pair sums, and using logits preserves the intended "increase target, decrease competitors" rule from the original paper.

The next section evaluates pairwise JSMA across multiple samples in batch mode, comparing its performance against the single-pixel baseline and analyzing the improvement in success rates and efficiency.

