# Demonstrating DeepFool

With the DeepFool algorithm fully implemented, we can execute end-to-end attacks that quantify the minimal perturbations needed to fool the classifier. How small can these perturbations be? The following demonstration measures this and exposes DeepFool's adaptive geometric strategy through concrete examples.

## Single-Image Attack Pipeline

Theory predicts DeepFool will find minimal perturbations through iterative boundary refinement. Does it actually work? We need to demonstrate the complete attack cycle on real data, measuring not just success but efficiency: iteration count, perturbation magnitude, and spatial distribution patterns. The demonstration will validate theoretical predictions against concrete results.

### Establishing the Attack Environment

Our first requirement is a trained target model and a clean sample to attack. Without these, we cannot generate adversarial perturbations or measure their characteristics. The model must be in evaluation mode to disable dropout, ensuring stable gradients during DeepFool's iterative refinement. We'll use a single-sample batch for detailed per-example analysis rather than batch aggregation.

Code: python

```python
# Load trained model
model_path = 'output/mnist_model.pth'
if os.path.exists(model_path):
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()
else:
    raise FileNotFoundError("Model not found.")

# Get single test sample
_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
dataiter = iter(test_loader)
image, true_label = next(dataiter)
image = image.to(device)

print(f"True label: {true_label.item()}")
```

The `model.eval()` call switches batch normalization to inference mode and disables dropout, which is necessary for DeepFool since gradient computations must be consistent across iterations. Dropout's random zeroing would create noisy, unstable gradients that violate the algorithm's assumption of smooth local linearity. The single-sample batch enables tracking exact iteration counts and perturbation evolution for this specific input rather than averaged batch statistics.

### Executing the Minimal Perturbation Attack

How far does this sample sit from the nearest decision boundary? We need to establish the baseline classification and confidence, then let DeepFool navigate toward the boundary. The baseline confidence indicates whether the sample lies deep within its class region (high confidence, many iterations expected) or near boundaries (low confidence, fast convergence). The attack execution returns the complete perturbation trajectory accumulated across all iterations.

Code: python

```python
# Baseline classification
with torch.no_grad():
    original_output = model(image)
    original_pred = original_output.argmax(dim=1).item()
    original_confidence = F.softmax(original_output, dim=1).max().item()

print(f"Original: class {original_pred} (confidence: {original_confidence:.3f})")

# Execute DeepFool attack
r_total, iterations, orig_label, pert_label, pert_image = deepfool(
    image, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
)

print(f"Attack: {orig_label} → {pert_label} in {iterations} iterations")
```

The `no_grad()` context disables gradient tracking for the baseline prediction since we're only reading the model's decision, not optimizing anything. The DeepFool call returns five values: `r_total` contains the complete accumulated perturbation vector, `iterations` indicates how many refinement steps were needed, `orig_label` and `pert_label` show the classification flip, and `pert_image` is the final adversarial example. Well-separated MNIST classes typically converge in 1-3 iterations, while ambiguous classes near decision boundaries may require 4-6 iterations. The 2% overshoot parameter adds `0.02 * perturbation` to ensure we actually cross the true non-linear boundary rather than just touching the linearized approximation, compensating for curvature the linear model cannot capture.

### Quantifying Attack Efficiency and Perturbation Properties

DeepFool claims to find minimal perturbations, but what does "minimal" actually mean in practice? We need multiple metrics to validate this claim. The L2 norm quantifies total perturbation energy (what DeepFool optimizes), while L∞ captures maximum pixel-wise change (what humans might notice). Comparing these exposes whether the perturbation spreads uniformly or concentrates on key features. The relative perturbation enables fair comparison across different image magnitudes, and the adversarial confidence indicates whether we barely crossed the boundary (confirming minimality) or penetrated deep into another class region (suggesting excess perturbation).

Code: python

```python
# Compute perturbation norms
perturbation_norm_l2 = torch.norm(r_total).item()
perturbation_norm_linf = torch.abs(r_total).max().item()
relative_perturbation = perturbation_norm_l2 / torch.norm(image).item()

# Evaluate adversarial confidence
with torch.no_grad():
    adv_output = model(pert_image)
    adv_confidence = F.softmax(adv_output, dim=1).max().item()

# Display results
print(f"\n=== Attack Results ===")
print(f"L2 norm: {perturbation_norm_l2:.4f}")
print(f"L∞ norm: {perturbation_norm_linf:.4f}")
print(f"Relative perturbation: {relative_perturbation:.2%}")
print(f"Original confidence: {original_confidence:.3f}")
print(f"Adversarial confidence: {adv_confidence:.3f}")
```

The L2/L∞ ratio demonstrates spatial concentration strategy. In our example, L2=7.72 and L∞=1.58. The maximum single-pixel change is 1.58, while total perturbation energy is 7.72. If the perturbation were perfectly uniform across all 784 pixels, we’d expect L∞≈L2/784≈7.72/28≈0.276. The observed 1.58 is nearly 6x larger than this uniform baseline, proving DeepFool concentrates modifications on strategically important pixels rather than distributing them uniformly. The relative perturbation formula ∥r∥2/∥x∥2 normalizes by image magnitude: a bright image with ∥x∥2=20 and a dark image with ∥x∥2=10 both showing 15% relative perturbation experienced proportionally equivalent modifications despite different absolute perturbation magnitudes. This digit ’7’ example shows 32.48% relative perturbation, which is higher than typical due to the significant structural change required to flip it to ’2’.

The adversarial confidence drop from 1.000 to 0.404 confirms minimal boundary crossing with slight overshoot. The model went from absolute certainty to below the 0.5 decision threshold. The confidence settling at 0.404 indicates the attack crossed the boundary and penetrated slightly into the '2' class region, as expected with the 2% overshoot parameter. This validates that DeepFool's iterative refinement successfully placed the adversarial example just beyond the decision surface, ensuring reliable misclassification while maintaining near-minimal perturbation magnitude.

## Exposing Spatial Attack Patterns

Metrics quantify perturbation magnitude, but where does DeepFool actually modify pixels? Theory suggests optimal geometric paths should target class-discriminative features along digit boundaries, not scatter noise uniformly. We need visual evidence. The challenge: minimal perturbations have such small magnitudes they're invisible when displayed at natural scale. We'll build a four-panel visualization showing the original image, an amplified perturbation pattern (normalized to full dynamic range for visibility), a magnitude heatmap showing intensity distribution, and the final adversarial result. This spatial analysis distinguishes DeepFool's targeted strategy from FGSM's uniform approach.

![Four-panel DeepFool visualization showing the original 7, the amplified perturbation pattern, the magnitude heatmap, and the resulting adversarial digit 2 with summary metrics.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/single_attack_visualization.png)

Code: python

```python
# Prepare images for visualization
original_img = mnist_denormalize(image.squeeze()).cpu().numpy()
adversarial_img = mnist_denormalize(pert_image.squeeze()).cpu().numpy()
perturbation = r_total.cpu().squeeze().numpy()

# Normalize perturbation for visibility (amplify minimal changes)
pert_display = perturbation - perturbation.min()
if pert_display.max() > 0:
    pert_display = pert_display / pert_display.max()

# Create four-panel visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)

# Panel 1: Original clean image
axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f"Original\nClass: {original_pred}",
                  color=HTB_GREEN, fontweight='bold')
axes[0].axis('off')

# Panel 2: Amplified perturbation pattern
axes[1].imshow(pert_display, cmap='inferno')
axes[1].set_title("Perturbation\n(amplified)",
                  color=NUGGET_YELLOW, fontweight='bold')
axes[1].axis('off')

# Panel 3: Perturbation magnitude heatmap
im = axes[2].imshow(np.abs(perturbation), cmap='viridis')
axes[2].set_title(f"Magnitude\nL2: {perturbation_norm_l2:.4f}",
                  color=AZURE, fontweight='bold')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Panel 4: Adversarial result
title_color = HTB_GREEN if pert_label != original_pred else MALWARE_RED
axes[3].imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
axes[3].set_title(f"Adversarial\nClass: {pert_label}",
                  color=title_color, fontweight='bold')
axes[3].axis('off')

# Summary metrics
metrics_text = (
    f"Iterations: {iterations}  |  "
    f"Relative pert: {relative_perturbation:.2%}  |  "
    f"Confidence: {original_confidence:.3f} → {adv_confidence:.3f}"
)
fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, color=WHITE)

plt.suptitle("DeepFool Attack Visualization", fontsize=14,
             color=HTB_GREEN, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

The normalization operation `(perturbation - min) / (max - min)` rescales the perturbation from its actual range (typically `[-0.5, 0.5]` for MNIST) to `[0, 1]` for visualization. Without this, the minimal perturbations would appear as uniform gray since matplotlib's colormap can't distinguish small variations. The `inferno` colormap maps 0 (minimal perturbation) to dark purple and 1 (maximum perturbation) to bright yellow, making spatial patterns visible. The `viridis` colormap for absolute magnitude uses the same principle but with green-yellow coloring better suited for intensity data.

Panel 2 demonstrates DeepFool's bidirectional strategy through color distribution. For our digit '7' to '2' attack, bright yellow clusters appear along the top and middle sections where the '7' stroke needs modification to resemble a '2'. Dark purple dominates background regions, showing minimal modification to irrelevant pixels. The asymmetry in color intensity (more brightening than darkening) reflects specific decision boundary geometry: transforming '7' into '2' requires adding features (the '2' curve) more than removing them. FGSM would show uniform yellow intensity across modified pixels regardless of importance. DeepFool's concentration pattern confirms the iterative gradient refinement successfully identified which pixels matter most for classification, allocating perturbation budget accordingly.

