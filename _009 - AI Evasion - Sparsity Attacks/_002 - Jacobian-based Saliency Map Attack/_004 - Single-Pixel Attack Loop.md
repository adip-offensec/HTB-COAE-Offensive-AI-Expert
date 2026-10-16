
# Single-Pixel Attack Loop

Now we demonstrate the complete attack by running iterations until success or budget exhaustion. The full attack requires careful orchestration of initialization, configuration, iteration logic, and results tracking. We'll build this incrementally to maintain visibility at each stage.

## Attack Configuration

To control attack behavior, we configure parameters governing perturbation magnitude, sparsity limits, and iteration budget. These settings determine the trade-off between attack subtlety and effectiveness:

Code: python

```python
# Reset for full attack
x_adv = x.clone().detach()
search_space = initialize_search_space(x.shape)

# Attack configuration
config = {
    'theta': 0.25,
    'gamma': 0.15,
    'max_iter': 100,
    'wrt': 'logits',
    'clip_min': 0.0,
    'clip_max': 1.0
}
```

The configuration dictionary centralizes all attack hyperparameters. The `theta` parameter controls step size (how much each pixel changes per modification), while `gamma` sets the feature budget as a fraction of total pixels. Setting `max_iter` prevents infinite loops when attacks stall. The `wrt` parameter specifies gradient computation with respect to logits (preserving independent target and competitor sensitivities), and clipping bounds ensure pixel values remain valid.

## Computing the Pixel Budget

The feature budget translates the abstract `gamma` fraction into a solid pixel count, establishing a hard limit on modifications:

Code: python

```python
num_features = int(np.prod(x.shape[1:]))
max_pixels = int(config['gamma'] * num_features)

print(f"Configuration:")
print(f"  Theta: {config['theta']}, Gamma: {config['gamma']}")
print(f"  Max pixels: {max_pixels} of {num_features}")
print(f"  Max iterations: {config['max_iter']}")
```

Output:

Code: txt

```txt
Configuration:
  Theta: 0.25, Gamma: 0.15
  Max pixels: 117 of 784
  Max iterations: 100
```

Computing the feature count requires flattening the spatial dimensions while ignoring the batch dimension. For MNIST with shape (1,1,28,28), `x.shape[1:]` yields (1,28,28), and `np.prod()` multiplies to get 1×28×28=784 pixels. Multiplying by `gamma=0.15` gives 0.15×784=117 pixels maximum. This budget represents 14.9% of the image, forcing the attack to select modifications strategically rather than modifying pixels arbitrarily.

## Understanding Parameter Trade-offs

Understanding the step size mechanics helps interpret attack behavior. Each selected pixel receives a perturbation of exactly ±θ. With `theta=0.25` on MNIST (where pixels range from 0 to 1), a black pixel at 0.0 jumps to 0.25 after one selection, then to 0.50 if selected again, requiring four selections to saturate at 1.0. This gradual progression demands more iterations but produces subtler visual artifacts. In contrast, `theta=1.0` saturates any pixel in a single selection, creating dramatic changes (0.0 → 1.0 instantly) that finish attacks quickly but risk detection. The feature budget implements hard sparsity constraints: setting `gamma=0.15` on MNIST calculates 0.15×784=117 maximum modifications, forcing the algorithm to choose wisely since the attack terminates once this limit is reached.

## Initializing Progress Tracking

We'll track key metrics at each iteration to monitor attack progression: iteration count, cumulative pixels modified, target class confidence, and saliency scores. This lets us analyze convergence behavior and identify when the attack stalls:

Code: python

```python
stats = {
    'iterations': [],
    'pixels_modified': [],
    'target_confidence': [],
    'saliency_scores': []
}

pixels_modified = 0

print(f"\nStarting attack: {original_class} → {target_class}")
print(f"Initial prediction: {model(x_adv).argmax(dim=1).item()}")
print(f"\n{'Iter':<6} {'Pixels':<8} {'Confidence':<12} {'Saliency':<12}")
print("="*46)
```

Output:

Code: txt

```txt
Starting attack: 7 → 2
Initial prediction: 7

Iter   Pixels   Confidence   Saliency
==============================================
```

Using lists as dict values creates a time-series database. Each list grows by one element per iteration, maintaining synchronized indices where element 0 across all lists corresponds to iteration 0. Why this parallel array pattern instead of a list of dicts? Appending to four separate lists is faster than creating a new dict per iteration, and extracting a complete time series (like all confidence values) requires just `stats['target_confidence']` instead of `[d['target_confidence'] for d in stats]`. The `pixels_modified` counter tracks cumulative L0 norm, starting at zero before any modifications. Column formatting uses fixed widths (6 characters for iteration number, 8 for pixel count, 12 for confidence and saliency), ensuring aligned output throughout the attack even as numbers change magnitude.

## Iteration Loop Structure

The attack loop implements a standard greedy optimization pattern: check termination conditions, compute gradients, select best modification, apply perturbation, update state, repeat. Each iteration makes irreversible progress toward the target class:

Code: python

```python
for iteration in range(config['max_iter']):
    # Check success condition
    if check_target_reached(x_adv, target_class, model):
        print(f"\n✓ Target reached at iteration {iteration}!")
        break

    # Check budget exhaustion
    if pixels_modified >= max_pixels:
        print(f"\n✗ Budget exhausted")
        break

    # Compute Jacobian and extract gradients
    jacobian = compute_jacobian_matrix(x_adv, model, num_classes=10, wrt=config['wrt'])
    alpha = extract_target_gradient(jacobian, target_class)
    beta = extract_other_gradients(jacobian, target_class)
    alpha_masked = apply_search_mask(alpha, search_space)
    beta_masked = apply_search_mask(beta, search_space)
```

Why check termination before computing gradients? Jacobian computation is expensive (10 backward passes, one per class), so we avoid this cost when success is already achieved or the budget exhausted. Checking success requires only a forward pass comparing the current prediction to the target class. Checking budget involves simple integer comparison of cumulative pixels against the limit. Both checks run in milliseconds, while Jacobian computation takes 10-15ms. Only when both pass do we proceed to the expensive computation, building the complete (10×784) gradient matrix.

Preparing gradients for saliency scoring happens through extraction and masking. We isolate the target class gradient (α) from row `target_class` of the Jacobian, then sum all other rows to get competitor gradients (β). Multiplying element-wise with the search space mask zeros out unavailable pixels, ensuring saliency scoring only considers modifiable pixels. Without masking, the algorithm might waste time scoring saturated or already-modified pixels that can’t contribute further.

## Saliency Scoring and Selection

With masked gradients ready, we score all pixels in both directions and select the winning modification:

Code: python

```python
    # Score both directions and select best
    inc_scores = score_increase_saliency(alpha_masked, beta_masked)
    dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
    pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

    # Check for exhausted search space
    if saliency <= 0:
        print(f"\n✗ No valid pixels remaining")
        break
```

Sign constraints filter candidates based on gradient direction. `score_increase_saliency` finds pixels where increasing helps: positive target gradient (raising this pixel boosts the target class) AND negative competitor gradient (raising this pixel suppresses competitors). `score_decrease_saliency` flips the logic: negative target gradient (lowering this pixel boosts the target class) AND positive competitor gradient (lowering this pixel suppresses competitors). Both return score arrays where invalid pixels receive zero, creating sparse saliency maps.

Tournament selection between directions happens in `select_best_direction`. We pit the highest increase score against the highest decrease score, returning the winner's pixel index, score value, and direction flag (`True` for increase, `False` for decrease). Zero saliency in both directions means no valid pixels remain (all are either saturated at clip bounds or violate sign constraints). This triggers early termination, avoiding wasted iterations that would make no progress.

## Perturbation Application and State Updates

With the best pixel selected, we apply the perturbation and update the search space to prevent reselection:

Code: python

```python
    # Apply perturbation
    x_adv = apply_single_pixel_perturbation(
        x_adv, pixel_idx, config['theta'], increase,
        config['clip_min'], config['clip_max']
    )

    # Update search space
    search_space[pixel_idx] = False
    search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)
```

Applying the perturbation requires flattening the image to access the selected pixel by index, adding or subtracting `theta` based on the direction flag, clamping to valid bounds, then reshaping to the original 4D structure. This modification is permanent. JSMA never undoes changes - each iteration makes irreversible progress.

Search space updates prevent wasted effort on unusable pixels through two-stage filtering. First, we immediately mask out the just-modified pixel via `search_space[pixel_idx] = False`. Why mask it even if it hasn't saturated? Because we've already extracted the maximum saliency from this pixel at this iteration. Reselecting it would waste the iteration since single-pixel JSMA modifies one pixel per step. Second, we scan the entire image for pixels at clip bounds (0.0 or 1.0), masking those out as well. Saturated pixels can't be modified further without violating clip constraints, so subsequent iterations should ignore them and focus on pixels with modification headroom.

## Progress Tracking and Display

After each modification, we record metrics and conditionally display progress:

Code: python

```python
    # Track metrics
    pixels_modified += 1
    confidence = compute_confidence(x_adv, target_class, model)

    stats['iterations'].append(iteration)
    stats['pixels_modified'].append(pixels_modified)
    stats['target_confidence'].append(confidence)
    stats['saliency_scores'].append(saliency)

    # Print progress
    if iteration % 5 == 0 or iteration < 3:
        print(f"{iteration:<6} {pixels_modified:<8} {confidence:<12.4f} {saliency:<12.6f}")
```

Output:

Code: txt

```txt
0      1        0.0000       0.641185
1      2        0.0000       0.501555
2      3        0.0000       0.363550
5      6        0.0000       0.300820
10     11       0.0000       0.343316
15     16       0.0000       0.230782
20     21       0.0000       0.175080
25     26       0.0000       0.095790
30     31       0.0000       0.071866
35     36       0.0000       0.050055
40     41       0.0000       0.050900
45     46       0.0000       0.038594
50     51       0.0000       0.037939
55     56       0.0000       0.029392
60     61       0.0001       0.043980
65     66       0.0001       0.033087
70     71       0.0001       0.022709
75     76       0.0001       0.011674
80     81       0.0001       0.014833
85     86       0.0002       0.008365
90     91       0.0002       0.031157
95     96       0.0002       0.014142
```

Incrementing the pixel counter maintains the cumulative L0 norm. Computing confidence requires a forward pass through the model with softmax to get probability distributions, extracting the target class probability. Appending to the stats lists builds the time-series record for post-attack analysis. The conditional print statement displays updates every 5 iterations plus the first 3, balancing informativeness against output volume. We use string formatting with fixed field widths to maintain column alignment: iteration left-aligned in 6 characters, pixels in 8, confidence in 12 with 4 decimal places, saliency in 12 with 6 decimal places.

## Final Results Evaluation

After the loop exits, we evaluate the final adversarial example and display comprehensive results:

Code: python

```python
final_success = check_target_reached(x_adv, target_class, model)
final_pred = model(x_adv).argmax(dim=1).item()

print("\n" + "="*46)
print(f"Final result: {'SUCCESS' if final_success else 'FAILED'}")
print(f"Final prediction: {final_pred}")
print(f"Pixels modified: {pixels_modified}/{max_pixels}")
print(f"Iterations: {len(stats['iterations'])}")
```

Output:

Code: txt

```txt
==============================================
Final result: FAILED
Final prediction: 7
Pixels modified: 100/117
Iterations: 100
==============================================
```

The final success check performs one last forward pass to confirm whether the adversarial example successfully forces misclassification to the target class. This redundant check (we already checked inside the loop) provides a definitive result independent of why the loop exited. Extracting the final prediction shows which class the model actually assigns, useful when attacks fail (prediction might be original class, target class, or a third unintended class). Displaying the pixel count against the budget reveals how close we came to exhausting resources. The iteration count comes from the stats list length rather than the loop variable, correctly accounting for early termination scenarios.

## Interpreting the Results

The attack failed after exhausting the maximum iteration limit, modifying 100 pixels out of 784 total (12.8%). The target confidence reached only 0.0002 (0.02%), never approaching the decision boundary needed for misclassification.

Notice how the saliency scores decline steadily from 0.641185 to 0.031157. This indicates the attack is making progressively less effective modifications as it exhausts high-value pixels. The confidence progression shows minimal growth throughout the attack, staying near zero for the first 50 iterations and only reaching 0.0002 by iteration 90. This pattern suggests single-pixel selection with `theta=0.25` struggles significantly on this particular sample. The attack identified pixels with positive saliency at each iteration but modifying them sequentially with small step sizes couldn't achieve the combination needed to cross the decision boundary.

This single-pixel baseline demonstrates JSMA's core iteration mechanics and saliency scoring. The canonical algorithm uses pairwise selection, which we implement in later sections. Batch evaluation in the next section characterizes typical single-pixel performance across multiple samples.

