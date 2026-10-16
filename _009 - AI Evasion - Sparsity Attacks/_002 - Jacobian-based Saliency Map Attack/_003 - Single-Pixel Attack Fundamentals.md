
# Single-Pixel Attack Fundamentals

We now build the complete single-pixel JSMA attack using the mathematical components from the previous section. Each component handles one step of the attack: applying perturbations, checking success, and tracking progress.

## Utility Functions

Before demonstrating the full attack, we need three utility functions that bridge between the mathematical saliency computation and actual image modification. These functions handle the mechanics of applying perturbations, checking whether we've achieved misclassification, and monitoring the target class confidence as the attack progresses.

### Applying Pixel Perturbations

Why can't we modify pixels directly in their 2D structure? PyTorch tensors support multi-dimensional indexing, but the saliency map returns flat pixel indices (0 to 783 for MNIST). We must flatten the image for indexed access, apply the perturbation, then restore the original 4D structure:

Code: python

```python
def apply_single_pixel_perturbation(x, pixel_idx, theta, increase, clip_min=0.0, clip_max=1.0):
    original_shape = x.shape
    x_flat = x.view(-1).clone()

    perturbation = theta if increase else -theta
    x_flat[pixel_idx] = torch.clamp(
        x_flat[pixel_idx] + perturbation,
        clip_min,
        clip_max
    )

    x_modified = x_flat.view(original_shape)
    return x_modified
```

How does PyTorch map flat indices back to 2D positions? `.view(-1)` reshapes the (1,1,28,28) tensor into a 784-element vector using row-major (C-style) ordering. Index 352 maps to channel 0, row 12, column 16 through the formula 0×28×28+12×28+16=352. Cloning happens first to prevent memory aliasing: views share memory with the original tensor, so modifying `x_flat[352]` without cloning would corrupt the input `x`, causing cascading errors in subsequent iterations.

Simple signed arithmetic handles perturbation direction: `+theta` for increase, `-theta` for decrease. But raw addition could push pixels outside valid bounds, creating values like -0.15 or 1.25 that violate the [0,1] image range. `torch.clamp()` enforces hard constraints at [clip_min,clip_max], saturating illegal values. A pixel at 0.9 receiving `+theta=0.25` would naively become 1.15, but clamping caps it at 1.0. Finally, `.view(original_shape)` restores the 4D structure using the same C-H-W ordering, ensuring flattened index 352 maps back to the correct spatial position.

### Checking Attack Success

To determine when to terminate the attack loop, we need a function that checks whether the current adversarial example successfully forces the model to predict our target class:

Code: python

```python
def check_target_reached(x, target_class, model):
    with torch.no_grad():
        logits = model(x)
        prediction = int(logits.argmax(dim=1).item())

    return prediction == target_class
```

Why run this check in inference mode? We only need a forward pass, no backpropagation. Wrapping in `torch.no_grad()` disables gradient tracking entirely, preventing PyTorch from building the computation graph. This saves memory (no graph storage for intermediate activations) and computation (no derivative calculations). Finding the predicted class requires `argmax(dim=1)`, which selects the highest logit across the class dimension and returns its index. Extracting with `.item()` converts the single-element tensor to a Python integer, enabling clean boolean comparison against `target_class` without tensor/scalar type mismatches.

### Computing Target Confidence

Beyond binary success/failure, we want to track how close we are to the decision boundary by monitoring the target class probability throughout the attack:

Code: python

```python
def compute_confidence(x, target_class, model):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence = float(probs[0, target_class].item())

    return confidence
```

How confident is the model in our target class? Tracking this quantifies attack progress beyond binary success/failure. Logits give us raw, unnormalized scores, but we need probabilities summing to 1.0 for meaningful confidence values. `F.softmax(logits, dim=1)` applies the softmax transformation ezi∑jezj, converting logits to a proper probability distribution across the class dimension. Indexing `[0, target_class]` extracts the target class probability from the first (and only) batch element. Watch this value during the attack: it starts near 0.0 for unlikely classes (maybe 0.001 for a digit the model strongly rejects) and should rise toward 1.0 as perturbations accumulate, crossing the decision boundary around 0.5 when misclassification occurs.

## Demonstrating a Single Iteration

Rather than hiding the attack mechanics inside a monolithic function, we'll execute one complete JSMA iteration to reveal the data flow explicitly. This demonstration shows exactly how each building block contributes to the attack cycle: gradient computation feeds into saliency scoring, which drives pixel selection, leading to perturbation application and search space updates.

We start by selecting a correctly classified sample and initializing the attack state:

Code: python

```python
# Get first correctly classified sample
for x_batch, y_batch in test_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        preds = model(x_batch).argmax(dim=1)

    for i in range(x_batch.size(0)):
        if preds[i].item() == y_batch[i].item():
            x = x_batch[i:i+1]
            original_class = int(y_batch[i].item())
            target_class = (original_class + 5) % 10
            break
    break

# Initialize attack state
x_adv = x.clone().detach()
theta = 0.25
clip_min, clip_max = 0.0, 1.0

# Initialize search space
num_features = int(np.prod(x.shape[1:]))
search_space = np.ones(num_features, dtype=bool)

print(f"Sample: digit {original_class}, target {target_class}")
print(f"Total features: {num_features}")
print(f"Initial target confidence: {compute_confidence(x_adv, target_class, model):.4f}")
```

Output:

Code: txt

```txt
Sample: digit 7, target 2
Total features: 784
Initial target confidence: 0.0000
```

Now we compute the Jacobian matrix, which requires 10 backward passes (one per class) to build the complete (10×784) sensitivity matrix. From this matrix, we extract the target class gradients (α) and sum of other class gradients (β), then apply the search space mask to zero out unavailable pixels:

Code: python

```python
jacobian = compute_jacobian_matrix(x_adv, model, num_classes=10, wrt='logits')
alpha = extract_target_gradient(jacobian, target_class)
beta = extract_other_gradients(jacobian, target_class)
alpha_masked = apply_search_mask(alpha, search_space)
beta_masked = apply_search_mask(beta, search_space)

print(f"Jacobian shape: {jacobian.shape}")
print(f"Target gradient range: [{alpha.min():.4f}, {alpha.max():.4f}]")
print(f"Other gradient range: [{beta.min():.4f}, {beta.max():.4f}]")
```

Output:

Code: txt

```txt
Jacobian shape: (10, 784)
Target gradient range: [-0.7783, 0.6853]
Other gradient range: [-0.7318, 0.8531]
```

Gradient sensitivities vary dramatically across pixels - some strongly influence the target class while barely affecting competitors, others do the opposite. To find the single best modification, we score all available pixels in both directions (increase and decrease) then select the winner:

Code: python

```python
inc_scores = score_increase_saliency(alpha_masked, beta_masked)
dec_scores = score_decrease_saliency(alpha_masked, beta_masked)
pixel_idx, saliency, increase = select_best_direction(inc_scores, dec_scores)

print(f"Increase: {(inc_scores > 0).sum()} valid pixels, max score {inc_scores.max():.6f}")
print(f"Decrease: {(dec_scores > 0).sum()} valid pixels, max score {dec_scores.max():.6f}")
print(f"Selected: pixel {pixel_idx}, saliency {saliency:.6f}, {'increase' if increase else 'decrease'}")
```

Output:

Code: txt

```txt
Increase: 330 valid pixels, max score 0.501555
Decrease: 349 valid pixels, max score 0.641185
Selected: pixel 398, saliency 0.641185, decrease
```

The scoring functions apply sign constraints to filter valid candidates: for increase, we need αj>0 (boosting target) AND βj<0 (suppressing competitors). Out of 784 pixels, only 330 satisfy these constraints. Decrease flips the logic: αj<0 AND βj>0, yielding 349 valid pixels. For valid pixels, saliency equals αj×|βj|, giving pixel 398 the maximum score of 0.641185 in the decrease direction. This multiplicative relationship means strong target influence combined with strong competitor suppression produces the highest scores.

With pixel 398 selected, we apply the perturbation using step size `theta=0.25`:

Code: python

```python
pixel_before = x_adv.flatten()[pixel_idx].item()
x_adv = apply_single_pixel_perturbation(x_adv, pixel_idx, theta, increase, clip_min, clip_max)
pixel_after = x_adv.flatten()[pixel_idx].item()

print(f"Pixel {pixel_idx}: {pixel_before:.4f} → {pixel_after:.4f}")
```

Output:

Code: txt

```txt
Pixel 398: 0.0000 → 0.0000
```

Pixel 398 started at 0.0 (black) and the decrease direction attempted to subtract `theta=0.25`, but `torch.clamp()` prevented it from going below 0.0, keeping it at 0.0. This demonstrates the boundary handling working correctly. After modifying the image, we update the search space to remove any pixels that have saturated at the boundaries (0.0 or 1.0), then check whether we've achieved misclassification:

Code: python

```python
search_space = remove_saturated_pixels(search_space, x_adv, clip_min, clip_max)
success = check_target_reached(x_adv, target_class, model)
confidence_new = compute_confidence(x_adv, target_class, model)

print(f"Modifiable pixels: {search_space.sum()}")
print(f"Target reached: {success}")
print(f"Target confidence: 0.0001 → {confidence_new:.4f}")
```

Output:

Code: txt

```txt
Modifiable pixels: 115
Target reached: False
Target confidence: 0.0000
```

Pixel 398 saturated at the minimum boundary (0.0), so it was removed from the search space along with any other boundary pixels, leaving 115 modifiable pixels. The target confidence remained at 0.0000 because this particular pixel modification didn't affect the model's prediction significantly. This single iteration demonstrates the complete cycle, but the attack needs many more iterations to accumulate enough changes for successful misclassification.

The next section assembles these components into the complete attack loop, orchestrating initialization, configuration, iteration logic, and results tracking to execute the full single-pixel JSMA attack.

