
# Pairwise Batch Evaluation

With the pairwise JSMA implementation complete, we now demonstrate its effectiveness across multiple samples and compare it directly with single-pixel JSMA. This section shows how pairwise selection exploits synergies and delivers superior performance in both success rates and efficiency.

## Batch Pairwise Attacks

Running pairwise attacks on the same 10 samples used for single-pixel JSMA enables direct comparison. We'll follow the same batch execution pattern from the single-pixel evaluation, but use pairwise selection instead of single-pixel.

### Initializing Pairwise Results Storage

To track pairwise attack performance, we prepare storage structures mirroring those from single-pixel evaluation:

Code: python

```python
print("Running pairwise attacks...")
print(f"{'#':<4} {'Orig→Tgt':<10} {'Result':<10} {'Pixels':<8} {'Iters':<8}")
print("="*46)

results_pairs = {
    'adversarial': [],
    'success': [],
    'pixels_modified': [],
    'iterations': []
}
```

Output:

Code: txt

```txt
Running pairwise attacks...
#    Orig→Tgt   Result     Pixels   Iters
==============================================
```

The `results_pairs` dictionary maintains the same structure as `results` from the single-pixel evaluation, enabling straightforward comparison between single-pixel and pairwise approaches. Using consistent naming conventions (appending `_pairs`) clarifies which results belong to which attack variant.

### Executing Pairwise Attacks on All Samples

We iterate through the collected samples, running a complete pairwise attack on each:

Code: python

```python
for idx in range(len(original_images)):
    x = original_images[idx]
    orig_class = original_labels[idx]
    tgt_class = target_labels[idx]

    # Initialize pairwise attack state
    x_adv = x.clone().detach()
    search_space = initialize_search_space(x.shape)
    pixels_mod = 0

    # Pairwise attack loop
    for iteration in range(config['max_iter']):
        # Termination checks
        if check_target_reached(x_adv, tgt_class, model):
            break
        if pixels_mod >= int(config['gamma'] * 784):
            break

        # Compute gradients
        jacobian = compute_jacobian_matrix(x_adv, model, 10, config['wrt'])
        alpha = extract_target_gradient(jacobian, tgt_class)
        beta = extract_other_gradients(jacobian, tgt_class)

        # Score both directions
        p_inc, q_inc, score_inc = compute_pairwise_saliency(
            alpha, beta, search_space, 'increase', config['top_k']
        )
        p_dec, q_dec, score_dec = compute_pairwise_saliency(
            alpha, beta, search_space, 'decrease', config['top_k']
        )

        # Select best pair
        if max(score_inc, score_dec) <= 0.0:
            break

        if score_inc >= score_dec:
            p, q, score, increase = p_inc, q_inc, score_inc, True
        else:
            p, q, score, increase = p_dec, q_dec, score_dec, False

        if p < 0 or q < 0:
            break

        # Apply perturbation and update
        x_adv = apply_pair_perturbation(
            x_adv, p, q, config['theta'], increase, 0.0, 1.0
        )

        search_space[p] = False
        search_space[q] = False
        search_space = remove_saturated_pixels(search_space, x_adv, 0.0, 1.0)
        pixels_mod += 2
```

The outer loop structure matches the single-pixel evaluation exactly: extract sample data, initialize fresh state, run attack loop. The inner loop replicates the pairwise attack pattern: check termination, compute Jacobian, score both directions, select best pair, apply perturbation, update state. Each attack runs independently with isolated state, preventing cross-contamination between samples. The pixel counter increments by 2 per iteration, reflecting pairwise modification.

### Recording and Displaying Pairwise Results

After each attack completes, we record metrics and display progress:

Code: python

```python
    # Record results
    success = check_target_reached(x_adv, tgt_class, model)

    results_pairs['adversarial'].append(x_adv)
    results_pairs['success'].append(success)
    results_pairs['pixels_modified'].append(pixels_mod)
    results_pairs['iterations'].append(iteration + 1)

    # Display progress
    status = "✓" if success else "✗"
    print(f"{idx+1:<4} {orig_class}→{tgt_class:<8} {status:<10} {pixels_mod:<8} {iteration+1:<8}")

print("="*46)
```

Output:

Code: txt

```txt
1    7→2        ✓          68       35
2    2→7        ✗          118      60
3    1→6        ✓          38       20
4    0→5        ✓          12       7
5    4→9        ✓          28       15
6    1→6        ✓          32       17
7    4→9        ✓          34       18
8    9→4        ✓          32       17
9    5→0        ✓          52       27
10   9→4        ✓          14       8
==============================================
```

Final success checking ensures consistent results independent of loop termination reason. Appending to result lists maintains parallel arrays where index positions correspond across all metrics. The display format matches single-pixel output exactly, enabling visual comparison. Status symbols provide immediate success/failure feedback.

Nine out of 10 samples were successfully attacked, with pixel counts ranging from 12 to 68 for successful attacks. Sample 2 (digit 2→7) exhausted the budget with 118 pixels without achieving misclassification, making it the only failure. Successful attacks completed in 7-35 iterations. For comparison, the single-pixel baseline with `theta=0.25` achieved 70% success (7/10) on these samples, while pairwise JSMA with `theta=1.0` succeeds on 90% of them.

## Comparing with Single-Pixel JSMA

Let's directly compare the efficiency of both variants:

Code: python

```python
print("\nEfficiency Comparison:")
print("="*60)
print(f"{'Sample':<8} {'Single-Pixel':<24} {'Pairwise':<24}")
print(f"{'':8} {'Iters':<8} {'Pixels':<8} {'':8} {'Iters':<8} {'Pixels':<8}")
print("="*60)

for idx in range(len(original_images)):
    single_iters = results['iterations'][idx]
    single_pixels = results['pixels_modified'][idx]
    pair_iters = results_pairs['iterations'][idx]
    pair_pixels = results_pairs['pixels_modified'][idx]

    print(f"{idx+1:<8} {single_iters:<8} {single_pixels:<8} {'':8} {pair_iters:<8} {pair_pixels:<8}")

# Compute statistics
mean_single_iters = np.mean(results['iterations'])
mean_pair_iters = np.mean(results_pairs['iterations'])
mean_single_pixels = np.mean(results['pixels_modified'])
mean_pair_pixels = np.mean(results_pairs['pixels_modified'])

print("="*60)
print(f"{'Mean':<8} {mean_single_iters:<8.1f} {mean_single_pixels:<8.1f} {'':8} {mean_pair_iters:<8.1f} {mean_pair_pixels:<8.1f}")
print("="*60)

iter_reduction = 100.0 * (mean_single_iters - mean_pair_iters) / mean_single_iters
pixel_difference = mean_single_pixels - mean_pair_pixels

print(f"\nIteration reduction: {iter_reduction:.1f}%")
print(f"Pixel difference: {pixel_difference:+.1f} pixels ({pixel_difference/mean_single_pixels*100:+.1f}%)")
```

Output:

Code: txt

```txt
Efficiency Comparison:
===========================================================
Sample   Single-Pixel             Pairwise
         Iters    Pixels            Iters    Pixels
===========================================================
1        100      100               35       68
2        100      100               60       118
3        68       67                20       38
4        34       33                7        12
5        59       58                15       28
6        93       92                17       32
7        61       60                18       34
8        87       86                17       32
9        100      100               27       52
10       41       40                8        14
===========================================================
Mean     74.3     73.6              22.4     42.8
===========================================================

Iteration reduction: 69.9%
```

The results demonstrate the improvement of pairwise selection over single-pixel approaches. Single-pixel attacks with `theta=0.25` achieved 70% success (7/10), with an average of 73.6 pixels modified. Pairwise attacks with `theta=1.0` improved to 90% success (9/10), using an average of 42.8 pixels (42% fewer). The pairwise approach required 69.9% fewer iterations on average (22.4 vs 74.3), demonstrating efficiency gains from simultaneous two-pixel modifications and better feature synergy exploitation. Sample 2 remains the only failure for pairwise (requiring 118 pixels without success), while samples 1 and 9 also failed for single-pixel. The transformation from 70% to 90% success with fewer pixels demonstrates pairwise JSMA's practical advantages.

The next section analyzes pair synergy, computational trade-offs, and configuration guidelines to deepen our understanding of why pairwise selection delivers such significant improvements.

