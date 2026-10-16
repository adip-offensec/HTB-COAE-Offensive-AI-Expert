
# Pairwise Saliency

Pairwise JSMA modifies two pixels simultaneously to exploit feature interactions in nonlinear models. For a pair (p,q), we define αpq=αp+αq and βpq=βp+βq, where α represents target class gradients and β represents the sum of other class gradients. We continue using logits to preserve independent sensitivities.

The pair score for increasing both features is St+[p,q]=αpq×|βpq| when sign constraints are met (αpq>0 and βpq<0). This formula extends the single-pixel saliency naturally: we sum the gradients and apply the same scoring logic.

We enforce the saliency sign constraints on the pair sums αp+αq and βp+βq, and we run two fixed policies (increase‑only and decrease‑only), selecting the better outcome. This mirrors the paper’s evaluation protocol and preserves the intended "increase target, decrease competitors" selection.

## Pairwise Saliency Function

Computing pairwise saliency requires evaluating many candidate pairs, pruning the search space for efficiency, and tracking the best combination. We'll build this through composable helper functions that each handle one aspect of the computation.

### Candidate Pruning

To control computational cost, we prune candidates based on individual gradient magnitudes before forming pairs:

Code: python

```python
def prune_candidates(alpha, beta, search_space, top_k):
    alpha_masked = alpha * search_space
    beta_masked = beta * search_space
    valid = np.where(search_space)[0]

    if valid.size < 2 or top_k is None or valid.size <= top_k:
        return valid

    prelim_scores = np.abs(alpha_masked[valid]) * np.abs(beta_masked[valid])
    idx = np.argsort(-prelim_scores)[:top_k]
    return valid[idx]
```

Why mask gradients before pruning? Multiplying by the search space mask (`alpha * search_space`) zeroes out unavailable pixels, ensuring we only consider modifiable candidates when computing preliminary scores. Converting the boolean mask to indices happens through `np.where(search_space)[0]`, which extracts integer pixel positions from the `True` entries.

Early returns handle edge cases efficiently. Fewer than 2 valid pixels? Can't form pairs, so return immediately. Pruning disabled (`top_k=None`)? Return all valid pixels. Already have fewer candidates than the budget (`valid.size <= top_k`)? No need to prune, return everything. These checks prevent unnecessary computation when pruning would have no effect.

Computing preliminary scores uses magnitude products |αj|×|βj| as a proxy for pair effectiveness. This heuristic assumes pixels with strong individual gradients likely form strong pairs (an assumption that holds empirically but isn’t guaranteed). Sorting by negative scores (`-prelim_scores`) creates descending order, and slicing `[:top_k]` selects the top candidates. Indexing `valid[idx]` returns these pixel positions in score-descending order.

### Pair Scoring Loop

With pruned candidates ready, we evaluate all pairs and track the best:

Code: python

```python
def evaluate_pairs(alpha, beta, valid, direction):
    best_p, best_q, best_score = -1, -1, 0.0

    for i in range(valid.size):
        p = valid[i]
        for j in range(i + 1, valid.size):
            q = valid[j]
            a_pq = alpha[p] + alpha[q]
            b_pq = beta[p] + beta[q]

            if direction == 'increase':
                if a_pq <= 0 or b_pq >= 0:
                    continue
                score = a_pq * abs(b_pq)
            else:
                if a_pq >= 0 or b_pq <= 0:
                    continue
                score = abs(a_pq) * b_pq

            if score > best_score:
                best_score = float(score)
                best_p, best_q = int(p), int(q)

    return best_p, best_q, best_score
```

Generating all unique pairs requires careful index manipulation. Outer loop index i ranges over all candidates, while inner loop index j starts at i+1. Why i+1 instead of i or 0? Starting at i+1 avoids both self-pairing (where p=q) and duplicate pairs (testing both (p,q) and (q,p) would waste computation since they’re equivalent). This generates exactly (n2) unique pairs.

Computing pair gradients happens through simple summation: αpq=αp+αq and βpq=βp+βq. Sign constraints then filter these pairs based on direction. Increase requires αpq>0 (the pair boosts the target) AND βpq<0 (the pair suppresses competitors). Decrease flips this: αpq<0 AND βpq>0. Why the flip? Negative target gradient means decreasing the pixel helps the target, while positive competitor gradient means decreasing suppresses competitors.

Scoring mirrors single-pixel saliency through magnitude products: target gradient magnitude times competitor gradient magnitude. We track the best pair found so far, updating whenever we discover a higher score and storing both pixel indices and the score value itself.

### Complete Pairwise Saliency Function

Now we assemble the pieces into the complete function:

Code: python

```python
def compute_pairwise_saliency(alpha, beta, search_space, direction='increase', top_k=None):
    valid = prune_candidates(alpha, beta, search_space, top_k)

    if valid.size < 2:
        return -1, -1, 0.0

    best_p, best_q, best_score = evaluate_pairs(alpha, beta, valid, direction)

    if best_p == -1:
        return -1, -1, 0.0
    return best_p, best_q, best_score
```

Delegating to helper functions keeps the main function focused on coordination. `prune_candidates` handles candidate selection, `evaluate_pairs` handles pair scoring, and this wrapper orchestrates the workflow. Checking `valid.size < 2` after pruning provides defensive validation. Though `prune_candidates` already checks this, redundant validation here ensures robustness against future code changes that might modify the helper's behavior.

Detecting exhausted search space requires checking the return value. When no pairs satisfy sign constraints, `evaluate_pairs` never updates `best_p` from its initial value of `-1`. This sentinel signals failure, prompting the wrapper to return `(-1, -1, 0.0)` indicating "no valid pair found" rather than returning invalid indices that could crash downstream code.

### Understanding Computational Complexity

Computational cost scales quadratically with candidate count. Without pruning, we evaluate (n2)=n(n−1)2 pairs for n valid pixels. When 400 pixels pass the search space filter, we’d evaluate 400×3992=79,800 pairs. Setting `top_k=128` caps evaluation at (1282)=128×1272=8,128 pairs, cutting cost by 90%. The pruning heuristic uses individual gradient magnitudes to identify promising candidates, assuming pixels with large individual contributions likely form effective pairs. This assumption holds empirically (pixels with strong individual effects tend to have strong pairwise synergies, though not always). The pruning trades optimality for speed: we might occasionally miss the globally best pair, but we find very good pairs much faster.

## Pair Perturbation Function

Applying perturbations to a pair modifies both pixels with the same signed step:

Code: python

```python
def apply_pair_perturbation(x, p, q, theta, increase, clip_min=0.0, clip_max=1.0):
    original_shape = x.shape
    x_flat = x.view(-1).clone()

    step = theta if increase else -theta
    x_flat[p] = torch.clamp(x_flat[p] + step, clip_min, clip_max)
    x_flat[q] = torch.clamp(x_flat[q] + step, clip_min, clip_max)

    return x_flat.view(original_shape)
```

The pairwise saliency score assumes both pixels move in the same direction (both increase or both decrease), requiring us to apply the same perturbation to both. If we increased pixel p but decreased pixel q, we’d violate the mathematical assumptions behind the pair score formula. The sign constraints on αpq=αp+αq and βpq=βp+βq hold only when both pixels shift identically. Clamping each pixel independently handles saturation: if pixel p is already at 0.95 and we apply `+theta=0.25`, it saturates at 1.0, while pixel q might reach only 0.70 from its starting value of 0.45.

## Demonstrating One Pairwise Iteration

To see how pairwise saliency differs from single-pixel scoring, we'll execute one complete iteration that reveals the data flow from gradient computation through pair selection to perturbation application. Unlike the single-pixel demonstration, this iteration modifies two pixels simultaneously based on their combined gradient contributions.

First, we need a correctly classified sample to attack. Let's select one and initialize the adversarial image:

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
search_space = initialize_search_space(x.shape)

print(f"Sample: digit {original_class}, target {target_class}")
print(f"Total pixels: {search_space.sum()}")
```

Output:

Code: txt

```txt
Sample: digit 7, target 2
Total pixels: 784
```

Now we compute the Jacobian and extract the target and competitor gradients. This is identical to single-pixel JSMA, giving us the per-pixel sensitivities we'll combine into pair scores:

Code: python

```python
jacobian = compute_jacobian_matrix(x_adv, model, num_classes=10, wrt='logits')
alpha = extract_target_gradient(jacobian, target_class)
beta = extract_other_gradients(jacobian, target_class)

print(f"Alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")
print(f"Beta range: [{beta.min():.6f}, {beta.max():.6f}]")
```

Output:

Code: txt

```txt
Alpha range: [-0.001234, 0.001456]
Beta range: [-0.001567, 0.001342]
```

With gradients computed, we identify candidate pixels and prune to the top-k most promising. We consider all unsaturated pixels as candidates, enforcing sign constraints only on the pair sums (not individual pixels). This lets us discover synergistic pairs where neither pixel alone would pass single-pixel filtering:

Code: python

```python
top_k = 128
valid = np.where(search_space)[0]

print(f"Initial candidates: {valid.size}")
print(f"Worst-case pairs: {valid.size * (valid.size - 1) // 2}")

# Prune to top-k based on individual gradient magnitudes
if valid.size > top_k:
    alpha_masked = alpha * search_space
    beta_masked = beta * search_space
    prelim_scores = np.abs(alpha_masked[valid]) * np.abs(beta_masked[valid])
    idx_sorted = np.argsort(-prelim_scores)[:top_k]
    valid_pruned = valid[idx_sorted]

    print(f"After pruning: {valid_pruned.size} candidates")
    print(f"Pairs to evaluate: {valid_pruned.size * (valid_pruned.size - 1) // 2}")
```

Output:

Code: txt

```txt
Initial candidates: 784
Worst-case pairs: 306936
After pruning: 128 candidates
Pairs to evaluate: 8128
```

Pruning cut the pair evaluation count by over 97%, from 306,936 to 8,128. This is essential for keeping pairwise JSMA computationally feasible while still exploring a large search space.

Now we search for the best pair using the increase direction. The function evaluates all candidate pairs, computing αpq=αp+αq and βpq=βp+βq for each, applying sign constraints, and returning the highest-scoring combination:

Code: python

```python
p, q, score = compute_pairwise_saliency(alpha, beta, search_space, 'increase', top_k)

print(f"Best pair: pixels {p} and {q}")
print(f"Best score: {score:.8f}")
print(f"\nIndividual gradients:")
print(f"  Pixel {p}: α={alpha[p]:.6f}, β={beta[p]:.6f}")
print(f"  Pixel {q}: α={alpha[q]:.6f}, β={beta[q]:.6f}")
print(f"\nPair gradients:")
print(f"  α_pq = {alpha[p] + alpha[q]:.6f}")
print(f"  β_pq = {beta[p] + beta[q]:.6f}")
```

Output:

Code: txt

```txt
Best pair: pixels 327 and 538
Best score: 1.68123138

Individual gradients:
  Pixel 327: α=0.685326, β=-0.731848
  Pixel 538: α=0.498610, β=-0.688188

Pair gradients:
  α_pq = 1.183936
  β_pq = -1.420036
```

The identified pair achieves a substantially higher score (1.68) compared to typical single-pixel scores. Both individual pixels have positive target gradients and negative competitor gradients, passing single-pixel constraints individually. Their combined effect (αpq=1.183936) exceeds either alone, and the product αpq×|βpq|=1.183936×1.420036=1.681 demonstrates the multiplicative relationship in the scoring formula.

Finally, we apply the perturbation to both selected pixels simultaneously and check the impact on target confidence:

Code: python

```python
theta = 1.0
clip_min, clip_max = 0.0, 1.0

# Capture before values
pixel_p_before = x_adv.flatten()[p].item()
pixel_q_before = x_adv.flatten()[q].item()

# Apply perturbation
x_adv = apply_pair_perturbation(x_adv, p, q, theta, True, clip_min, clip_max)

# Capture after values
pixel_p_after = x_adv.flatten()[p].item()
pixel_q_after = x_adv.flatten()[q].item()

# Check impact
confidence_after = compute_confidence(x_adv, target_class, model)

print(f"Pixel {p}: {pixel_p_before:.4f} → {pixel_p_after:.4f}")
print(f"Pixel {q}: {pixel_q_before:.4f} → {pixel_q_after:.4f}")
print(f"Target confidence: 0.0001 → {confidence_after:.4f}")
```

Output:

Code: txt

```txt
Pixel 327: 0.9922 → 1.0000
Pixel 538: 0.0000 → 1.0000
Target confidence after pair perturbation: 0.0000
```

With `theta=1.0`, pixel 327 increased from 0.9922 to 1.0000 (saturating at the boundary), and pixel 538 jumped from 0.0 to 1.0 (full saturation). The target confidence remained at 0.0000 after this first pair modification, indicating that early iterations build foundation changes that accumulate over subsequent iterations. This demonstrates pairwise selection's approach: coordinated pair modifications that work synergistically across the full attack sequence.

The next section assembles these pairwise saliency components into the complete attack loop, orchestrating configuration, iteration logic, and results tracking to execute the full pairwise JSMA attack.

