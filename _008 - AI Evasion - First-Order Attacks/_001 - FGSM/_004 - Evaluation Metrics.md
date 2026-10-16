
# Evaluation Metrics

With the core FGSM attack implemented, we now build metrics to evaluate its effectiveness. Flip rate alone doesn’t tell the full story. What happens to model confidence? How much did perturbations actually change the image in L2 terms? This section implements an evaluation function that tracks accuracy, success rate, confidence drop, and perturbation norms, giving a complete picture of attack impact.

## What Metrics Do We Need?

Evaluating an adversarial attack requires looking beyond whether predictions flip. We need to understand the full impact on model behavior. Accuracy metrics tell us how many samples the model classifies correctly before and after perturbation, establishing the baseline and measuring degradation. Success rate specifically measures what percentage of originally correct predictions the attack manages to flip, isolating attack effectiveness from model quality. Confidence metrics reveal how certain the model is in its predictions, showing whether the attack creates genuine confusion or just barely pushes samples across decision boundaries. Perturbation norms quantify the distortion introduced, measuring both average L2 distance per sample and maximum L∞ change to verify we respect the epsilon budget.

## Building the Evaluation Function

First, we need the model's predictions and confidence scores for both clean and adversarial inputs. Running both batches through the model yields logits, which we convert to probabilities using softmax and extract predicted classes using argmax:

Code: python

```python
from typing import Dict

def evaluate_attack(model: nn.Module,
                   clean_images: Tensor,
                   adversarial_images: Tensor,
                   true_labels: Tensor) -> Dict[str, float]:
    """Compute accuracy, success rate, confidence shift, and norms.

    Args:
        model: Evaluated classifier in evaluation mode
        clean_images: Clean inputs in the model's expected domain (e.g., normalized MNIST)
        adversarial_images: Adversarial counterparts in the same domain as `clean_images`
        true_labels: Ground-truth labels

    Returns:
        Dict[str, float]: Aggregated metrics summarizing attack impact
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_images)
        adv_logits = model(adversarial_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)
```

The `torch.no_grad()` context disables gradient tracking since we're only evaluating, not training. For a batch of 128 images with 10 classes, `clean_logits` has shape `[128, 10]`, `clean_probs` has the same shape with rows summing to 1.0, and `clean_pred` has shape `[128]` containing class indices.

### Measuring Accuracy and Success Rate

Next, we compute correctness masks and derive the attack success rate. Comparing predictions to true labels yields boolean tensors indicating which samples are classified correctly. The attack success rate focuses specifically on samples that were originally correct but became incorrect after perturbation:

Code: python

```python
        clean_correct = (clean_pred == true_labels)
        adv_correct = (adv_pred == true_labels)

        originally_correct = clean_correct
        flipped = (~adv_correct) & originally_correct
```

The `flipped` mask uses logical operations to identify samples where `clean_correct` is `True` but `adv_correct` is `False`. If 125 out of 128 samples were originally correct and 80 flipped, the success rate is 80/125=0.64.

### Extracting Confidence Metrics

Confidence metrics reveal how certain the model is in its predictions. We extract the probability the model assigns to the true class for each sample, both before and after perturbation. The `gather` operation selects specific probability values based on the true label indices:

Code: python

```python
        conf_clean = clean_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        conf_adv = adv_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
```

How does `gather` work? If `true_labels=[3, 7]` and `clean_probs` has shape `[2, 10]`, then `true_labels.view(-1, 1)` reshapes to `[[3], [7]]`. The `gather(1, ...)` operation selects column 3 from row 0 and column 7 from row 1, extracting the probabilities for the true classes. The `squeeze(1)` removes the extra dimension, yielding a 1D tensor. A confidence drop from 0.92 to 0.31 indicates the attack eroded the model's certainty by 61 percentage points, even if the prediction didn't flip.

### Computing Perturbation Norms

Finally, we measure the magnitude of perturbations introduced by the attack. The L2 norm quantifies the Euclidean distance between clean and adversarial images (measuring overall distortion), while the L∞ norm captures the maximum change to any single pixel (verifying we respect the epsilon budget):

Code: python

```python
        l2 = (adversarial_images - clean_images).view(clean_images.size(0), -1).norm(p=2, dim=1)
        linf = (adversarial_images - clean_images).abs().amax()

        return {
            "clean_accuracy": clean_correct.float().mean().item(),
            "adversarial_accuracy": adv_correct.float().mean().item(),
            # Success rate among originally correct samples only
            "attack_success_rate": (
                flipped.float().sum() / originally_correct.float().sum().clamp_min(1.0)
            ).item(),
            "avg_clean_confidence": conf_clean.mean().item(),
            "avg_adv_confidence": conf_adv.mean().item(),
            "avg_confidence_drop": (conf_clean - conf_adv).mean().item(),
            "avg_l2_perturbation": l2.mean().item(),
            "max_linf_perturbation": linf.item(),
        }
```

The `.view(clean_images.size(0), -1)` flattens each image to a 1D vector while preserving the batch dimension, enabling per-sample L2 norm computation. For MNIST images with shape `[128, 1, 28, 28]`, this becomes `[128, 784]`. The `norm(p=2, dim=1)` computes the L2 norm along dimension 1, yielding a tensor of shape `[128]` with one norm value per image. The L∞ norm uses `.amax()` without specifying dimensions, returning the single largest absolute perturbation across the entire batch.

## Applying the Metrics

Applying these metrics to the batch from Core Implementation:

Code: python

```python
# Assume images, labels, x_adv from the Core Implementation section
metrics = evaluate_attack(model, images, x_adv, labels)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```

Expected output:

Code: txt

```txt
clean_accuracy: 0.9766
adversarial_accuracy: 0.3203
attack_success_rate: 0.6797
avg_clean_confidence: 0.9824
avg_adv_confidence: 0.3891
avg_confidence_drop: 0.5933
avg_l2_perturbation: 10.8451
max_linf_perturbation: 0.8000
```

The metrics show that FGSM with `epsilon=0.8` (in normalized space, approximately `epsilon=0.25` in pixel space) successfully flips 67.97% of originally correct predictions, dropping average confidence from 98.24% to 38.91%. The `max_linf_perturbation` confirms the attack respects the epsilon bound. The L2 perturbation of approximately 10.8 reflects the normalized space magnitude, which corresponds to subtle visual changes when denormalized for display.

