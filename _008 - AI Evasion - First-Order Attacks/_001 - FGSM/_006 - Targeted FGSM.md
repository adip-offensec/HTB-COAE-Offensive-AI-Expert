
Targeted FGSM changes the objective from reducing confidence in the true class to increasing confidence in a specific target class. With cross-entropy, untargeted FGSM ascends ℒ(θ,x,y); targeted FGSM descends ℒ(θ,x,yt), equivalent to ascending −ℒ(θ,x,yt).

The implementation has two precise differences from untargeted. First, the gradient uses target label yt instead of true label. Second, the update sign flips so the step reduces loss for yt rather than increasing loss for y. The L∞ budget and clipping remain unchanged.

Because targeted FGSM must steer predictions toward a particular class (not just away from the current one), it often requires slightly larger ϵ or benefits from early stopping once the model predicts the target.

## Targeted Attack Example

The implementation forces a 1→7 flip by supplying the target label in the loss and reversing the update direction. We need to find a digit 1 in the test set, then search for the minimal epsilon that successfully forces the model to predict 7. The search uses progressively larger epsilon values and terminates early once any candidate succeeds:

### Setup and Configuration

To find the minimal perturbation budget, we'll test epsilon values in ascending order and track the first successful configuration:

Code: python

```python
eps_candidates = [0.5, 0.8, 1.0]
success_image, success_label, success_eps = None, None, None
```

The `eps_candidates` list specifies epsilon values in normalized space, starting from a conservative 0.5 and increasing to 1.0. The three `None` values track results: `success_image` stores the original digit 1, `success_label` holds its true label, and `success_eps` records which epsilon value succeeded. Starting all at `None` lets us detect failure by checking if they remain unset after the search completes.

### Finding a Candidate Digit

We need to locate a digit 1 from the test set that the model correctly classifies, ensuring we have a valid baseline for the targeted attack:

Code: python

```python
model.eval()
candidate, candidate_label = None, None

for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)
    match_indices = (yb == 1).nonzero(as_tuple=True)[0]
    if len(match_indices) == 0:
        continue

    # Check predictions for all digit 1s in this batch
    with torch.no_grad():
        preds = model(xb[match_indices]).argmax(dim=1)
        correct_mask = (preds == 1)
        if correct_mask.any():
            # Take first correctly classified digit 1
            local_idx = correct_mask.nonzero(as_tuple=True)[0][0].item()
            idx = match_indices[local_idx].item()
            candidate = xb[idx]
            candidate_label = yb[idx]
            break

if candidate is None:
    raise RuntimeError("Could not find a correctly classified digit 1 in test set")
```

The loop iterates over batches from `test_loader`, yielding tensors `xb` and `yb` with shapes `[128, 1, 28, 28]` and `[128]` respectively. The expression `(yb == 1)` creates a boolean tensor where `True` marks positions containing label 1. If a batch has labels `[3, 1, 7, 1, 2]`, the comparison produces `[False, True, False, True, False]`. The `nonzero(as_tuple=True)[0]` operation extracts the indices of `True` values as a 1D tensor, yielding `[1, 3]` for this example.

Before selecting a candidate, we verify the model actually predicts it as class 1 by running inference on all digit 1s in the batch with `model(xb[match_indices])`. The `correct_mask = (preds == 1)` identifies which ones are correctly classified. We only proceed if at least one is correct, using `correct_mask.any()`. The double indexing `match_indices[local_idx]` maps from the filtered subset back to the original batch position. Setting `model.eval()` before the loop ensures batch normalization and dropout behave correctly during inference. The final check `if candidate is None` catches the edge case where no correctly classified digit 1 exists in the entire test set.

### Testing Epsilon Values

Now we run targeted FGSM with each epsilon candidate and check if the attack forces prediction to class 7:

Code: python

```python
target_label = torch.tensor([7], device=device)

for eps_try in eps_candidates:
    x_adv = fgsm_attack(
        model,
        candidate.unsqueeze(0),
        target_label,
        epsilon=eps_try,
        targeted=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        success_image = candidate
        success_label = candidate_label
        success_eps = eps_try
        break
```

The loop tests epsilon values in ascending order. The `fgsm_attack` call requires careful tensor shaping. The `candidate` has shape `[1, 28, 28]` after indexing from the batch, so `unsqueeze(0)` adds a batch dimension to produce `[1, 1, 28, 28]`, matching the model's expected input format. The `target_label` is created once before the loop as `torch.tensor([7], device=device)`, producing a tensor of shape `[1]` already on the correct device. The `targeted=True` flag tells `fgsm_attack` to reverse the gradient direction, minimizing loss toward class 7 instead of maximizing loss for the true class.

After generating the adversarial, we check the model's prediction with `argmax(dim=1).item()`. If `pred == 7`, the attack succeeded at this epsilon, so we capture the configuration and break immediately. This early termination avoids testing larger epsilon values once we find the minimal budget that works.

### Handling Failure and Visualization

If no epsilon succeeded, we raise an error. Otherwise, visualize the successful attack:

Code: python

```python
if success_image is None:
    raise RuntimeError("Targeted FGSM did not achieve 1 -> 7 within the tested epsilons.")

_ = visualize_fgsm_attack(
    model,
    success_image.detach().cpu(),
    success_label.detach().cpu(),
    success_eps,
    targeted=True,
    target_class=7,
)
```

The `success_image is None` check detects complete failure across all epsilon candidates. If all tested values failed to force the target prediction, the error message helps debug by indicating that larger epsilon values or different hyperparameters may be needed. On success, we move tensors back to CPU with `.detach().cpu()` before passing to the visualization function, which expects CPU tensors for matplotlib rendering.

Expected output:

Code: txt

```txt
epsilon=0.50 -> predicted 1
epsilon=0.80 -> predicted 7
```

The output shows that `epsilon=0.5` in normalized space fails to flip the prediction, while increasing to `epsilon=0.8` succeeds and forces class 7. Targeted attacks often need larger budgets than untargeted ones because they must push toward a particular class, but the exact threshold depends on the specific sample and trained weights.

![Targeted FGSM example shifts a digit 1 toward class 7 with an amplified perturbation map and probability bars showing partial target confidence.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/fgsm_targeted.png)

