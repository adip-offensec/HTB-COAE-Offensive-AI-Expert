
# Saliency and Search Space

Saliency scoring translates raw gradients into pixel importance rankings through sign constraints and magnitude products. Managing the search space ensures we don't waste iterations on saturated pixels or violate boundary constraints. Together, these components guide JSMA's iterative selection process.

## Saliency Scoring

The saliency score determines which pixel to modify. We need to score both increase and decrease directions separately, then choose the higher-scoring option. The scoring formula is score=|α|×|β| when sign constraints are met, otherwise zero.

To find pixels where raising the value boosts our target while suppressing competitors, let's implement scoring for the increase direction first:

Code: python

```python
def score_increase_saliency(target_grad, other_grad):
    increase_mask = (target_grad > 0) & (other_grad < 0)
    scores = target_grad * np.abs(other_grad) * increase_mask
    return scores
```

Building on the gradient interpretation from the previous section, sign constraints filter candidates through Boolean logic. The expression `(target_grad > 0) & (other_grad < 0)` creates a mask that’s `True` only where both conditions hold. For pixel j, we need ∂Ft∂xj>0 (increasing this pixel helps our target) AND ∑i≠t∂Fi∂xj<0 (increasing this pixel hurts competitors). Multiplying by this mask zeroes out invalid pixels. For valid pixels, the score becomes αj×|βj|. Notice we use `target_grad` directly (already positive by constraint) but take `np.abs(other_grad)` because `other_grad` is negative, and we want the magnitude.

Now we implement the decrease direction with flipped sign requirements:

Code: python

```python
def score_decrease_saliency(target_grad, other_grad):
    decrease_mask = (target_grad < 0) & (other_grad > 0)
    scores = np.abs(target_grad) * other_grad * decrease_mask
    return scores
```

Decreasing reverses the gradient interpretation. A negative target gradient means the target class decreases when we increase the pixel, so decreasing the pixel increases the target class. Similarly, a positive "others" gradient means competitors increase when we increase the pixel, so decreasing the pixel decreases competitors. Both effects align with our goal. The mask becomes `(target_grad < 0) & (other_grad > 0)`, filtering for exactly this scenario. The score formula uses `np.abs(target_grad)` (converting negative to positive magnitude) times `other_grad` (already positive by constraint).

Example:

Code: python

```python
# Gradients for 6 features
alpha = np.array([ 0.6, -0.3,  0.4,  0.1, -0.5,  0.2])
beta  = np.array([-0.2,  0.4, -0.5,  0.3,  0.6, -0.1])

inc_scores = score_increase_saliency(alpha, beta)
dec_scores = score_decrease_saliency(alpha, beta)

print("Feature | α     β    | Inc Score | Dec Score")
print("--------|-----------|-----------|----------")
for i in range(len(alpha)):
    print(f"   {i}    | {alpha[i]:5.1f} {beta[i]:5.1f} |  {inc_scores[i]:7.3f}  |  {dec_scores[i]:7.3f}")
```

Output:

Code: txt

```txt
Feature | α     β    | Inc Score | Dec Score
--------|-----------|-----------|----------
   0    |   0.6  -0.2 |    0.120  |   -0.000
   1    |  -0.3   0.4 |   -0.000  |    0.120
   2    |   0.4  -0.5 |    0.200  |   -0.000
   3    |   0.1   0.3 |    0.000  |    0.000
   4    |  -0.5   0.6 |   -0.000  |    0.300
   5    |   0.2  -0.1 |    0.020  |   -0.000
```

Feature 0 is valid for increase (α>0, β<0) with score 0.6×0.2=0.120. Feature 1 is valid for decrease (α<0, β>0) with score 0.3×0.4=0.120. Feature 2 achieves the best increase score of 0.200. Feature 3 is invalid for both directions, violating sign constraints. Feature 4 achieves the best overall score of 0.300 in the decrease direction.

Having implemented both direction scoring functions, we now need to select the winning pixel and direction:

Code: python

```python
def select_best_direction(inc_scores, dec_scores):
    max_inc_idx = int(np.argmax(inc_scores))
    max_dec_idx = int(np.argmax(dec_scores))

    max_inc_score = float(inc_scores[max_inc_idx])
    max_dec_score = float(dec_scores[max_dec_idx])

    if max_inc_score > max_dec_score:
        return max_inc_idx, max_inc_score, True
    else:
        return max_dec_idx, max_dec_score, False
```

We pit both directions against each other through tournament selection. `np.argmax()` finds the highest-scoring pixel in each direction's score array, returning the index. Extracting the actual scores at these indices lets us compare magnitudes directly. The winner gets returned as a tuple: pixel index, score value, and direction flag (`True` for increase, `False` for decrease). What happens when all scores are zero? `np.argmax()` returns index 0 by convention, and `max_inc_score` becomes 0.0. The calling code checks for zero scores and terminates the attack when no valid modifications exist.

Continuing the previous example:

Code: python

```python
pixel_idx, score, increase = select_best_direction(inc_scores, dec_scores)
direction = "increase" if increase else "decrease"
print(f"Selected: pixel {pixel_idx}, score {score:.3f}, {direction}")
```

Output:

Code: txt

```txt
Selected: pixel 4, score 0.300, decrease
```

Feature 4 wins with the highest score (0.300) in the decrease direction, so we would decrease that pixel's value.

## Search Space Management

The search space tracks which features remain modifiable through a boolean mask. Starting with all pixels available, we progressively remove features as we modify them or they saturate at boundaries. This prevents wasting iterations on unchangeable pixels.

Let's implement the initialization function:

Code: python

```python
def initialize_search_space(shape):
    num_features = int(np.prod(shape[1:]))
    return np.ones(num_features, dtype=bool)
```

Extracting the feature count from a batch input with shape (B,C,H,W) requires computing the flattened size while ignoring the batch dimension. `shape[1:]` slices to `(C, H, W)`, and `np.prod()` multiplies these together. For MNIST with shape (1,1,28,28), this yields 1×28×28=784. Creating a boolean array with `np.ones(..., dtype=bool)` initializes every pixel to `True` (available). Why boolean instead of integers? Boolean arrays consume 1 byte per element versus 4-8 bytes for integers, saving memory. They also make intent explicit: this is a binary mask, not a count.

Example:

Code: python

```python
# MNIST image shape
shape = (1, 1, 28, 28)
search_space = initialize_search_space(shape)
print(f"Search space shape: {search_space.shape}")
print(f"Initial modifiable pixels: {search_space.sum()}")
```

Output:

Code: txt

```txt
Search space shape: (784,)
Initial modifiable pixels: 784
```

All 784 pixels are available at the start. As we modify pixels or they saturate, we'll set their mask values to false.

Saturation occurs when pixels reach the minimum or maximum allowed values. Once saturated, further modifications in that direction have no effect, making these pixels useless for the attack. Let's detect and remove them:

Code: python

```python
def remove_saturated_pixels(search_space, x, clip_min=0.0, clip_max=1.0, epsilon=1e-6):
    x_flat = x.detach().cpu().numpy().flatten()

    saturated_min = (x_flat <= clip_min + epsilon)
    saturated_max = (x_flat >= clip_max - epsilon)
    saturated = saturated_min | saturated_max

    updated_mask = search_space & ~saturated
    return updated_mask
```

Epsilon tolerance for boundary detection accounts for floating-point rounding errors. A pixel at exactly 0.0 might be stored as 0.0000001 or -0.0000001 due to previous operations. Checking `x_flat <= clip_min + epsilon` catches pixels within 10−6 of the minimum, treating them as effectively saturated. We combine minimum and maximum saturation using the `|` operator into a single mask: any pixel saturated in either direction gets marked `True`. Finally, `search_space & ~saturated` performs Boolean intersection between current availability and non-saturated pixels. The `~` inverts the saturation mask (`True→False`, `False→True`), so we keep pixels that are both currently available AND not saturated.

Example:

Code: python

```python
# Toy image with some saturated pixels
x_toy = torch.tensor([[[[0.0, 0.3], [0.95, 1.0]]]])  # 4 pixels
mask = np.array([True, True, True, True])

updated_mask = remove_saturated_pixels(mask, x_toy, clip_min=0.0, clip_max=1.0)
print(f"Original mask:  {mask}")
print(f"Pixel values:   {x_toy.flatten().numpy()}")
print(f"Updated mask:   {updated_mask}")
print(f"Remaining pixels: {updated_mask.sum()}/4")
```

Output:

Code: txt

```txt
Original mask:  [ True  True  True  True]
Pixel values:   [0.   0.3  0.95 1.  ]
Updated mask:   [False  True  True False]
Remaining pixels: 2/4
```

Pixels 0 and 3 are saturated (at 0.0 and 1.0 respectively) and removed from the search space. Only pixels 1 and 2 remain modifiable.

