
# Attack Visualizations

What does 95% success actually mean? Numbers tell part of the story. The attack succeeded on 19 out of 20 examples with average L2 distortion of 2.36. But what do these adversarial examples actually look like? How does the perturbation distribute across pixels? Which examples required larger distortions?

Statistics provide the numbers. Visualizations reveal the story. Summary metrics like "average L2 distortion of 2.36" compress 20 examples into a single value. Visualizations expose the distribution: do all examples cluster near 2.36, or do some need minimal perturbations while others require massive changes? Bar charts reveal these patterns. Histograms show the full distribution shape.

This section develops visualization blocks showing the attack transformation process and distortion characteristics. We'll examine original and adversarial images side by side, then analyze how perturbation magnitudes distribute across examples. All visualizations follow the HTB theme using colors and styling from the `htb_ai_library`.

## Attack Process Visualization

The most direct way to understand adversarial examples is seeing them. This visualization displays original images, adversarial images, and the perturbations that transform one into the other:

Code: python

```python
print("\nGenerating visualizations...")

# Visualization: Attack process (original → perturbation → adversarial)
print("Creating attack process visualization...")

fig, axes = plt.subplots(3, 10, figsize=(20, 6))

# Show first 10 examples
num_display = min(10, batch_size)

for i in range(num_display):
    # Original image
    orig_img = original_images[i].detach().cpu().squeeze()
    axes[0, i].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"True: {attack_targets[i].item()}", fontsize=10)

    # Perturbation (amplified for visibility)
    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()
    pert_display = pert * 10  # Amplify by 10x for visibility
    axes[1, i].imshow(pert_display, cmap="seismic", vmin=-1, vmax=1)
    axes[1, i].axis("off")
    axes[1, i].set_title("Perturbation", fontsize=10)

    # Adversarial image
    adv_img = best_adv[i].detach().cpu().squeeze()
    axes[2, i].imshow(adv_img, cmap="gray", vmin=0, vmax=1)
    axes[2, i].axis("off")
    axes[2, i].set_title(f"Pred: {adv_predictions[i].item()}", fontsize=10, color=MALWARE_RED)

# Row labels
fig.text(0.02, 0.80, "Original", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.50, "Perturbation\n(10× amplified)", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.20, "Adversarial", rotation=90, fontsize=14, weight="bold", ha="center", va="center")

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(output_dir / "ead_attack_process.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_attack_process.png")
```

We visualize the three-stage attack transformation using a 3×10 grid layout. Original images appear in the top row with their true labels. Perturbations occupy the middle row, amplified 10× for visibility since actual perturbations are imperceptible at normal scale. Adversarial images fill the bottom row with predicted labels, colored red to emphasize misclassification.

Amplifying perturbations by 10× makes their subtle patterns visible while preserving structure. We use the seismic colormap where blue represents negative changes and red represents positive changes, with white indicating unchanged pixels. This color scheme reveals exactly where and how the attack modified each image.

![Grid of 10 MNIST digits showing original, 10x‑amplified perturbations, and adversarial results with predicted labels](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/ead_attack_process.png)

The visualization exposes the attack's precision. Original digits remain perfectly recognizable in the top row. The middle row reveals perturbations concentrated along edges and stroke boundaries rather than scattered randomly. Notice how Example 1 shows modifications clustering around the digit's curves, while Example 4 targets the loop region. The adversarial images in the bottom row look nearly identical to originals, yet the model misclassifies them. A true 7 becomes predicted as 2, a true 0 becomes predicted as 6. The human eye struggles to detect any difference, but the model's decision boundary has been crossed.

## Distortion Distribution Analysis

Numbers provide precision. Distributions reveal patterns. Mean distortion tells us average perturbation magnitude, but what about the spread? Do all examples cluster tightly around the mean, or does variability suggest different robustness levels? Histograms expose these distributions, showing whether the attack behaves uniformly or reveals heterogeneous robustness across examples.

To understand how different distance metrics characterize our perturbations, we’ll create histograms for each norm. The four metrics (L1, L2, L∞, and Elastic-Net) capture different aspects of perturbation structure, and their distributions tell complementary stories about attack behavior:

Code: python

```python
# Visualization 2: Distortion distributions for L1, L2, L∞, Elastic
print("Creating distortion analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate distortion metrics
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
linf_values = linf_dist.detach().cpu().numpy()
elastic_values = elastic_dist.detach().cpu().numpy()

# L1 distribution
axes[0, 0].set_title(r"$L_1$ Distortion Distribution", fontsize=14)
axes[0, 0].set_xlabel(r"$L_1$ Distance")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].hist(l1_values, bins=15, color=AZURE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 0].axvline(l1_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l1_values.mean():.2f}")
axes[0, 0].legend(frameon=False, fontsize=10)

# L2 distribution
axes[0, 1].set_title(r"Squared $L_2$ Distortion Distribution", fontsize=14)
axes[0, 1].set_xlabel(r"Squared $L_2$ Distance")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].hist(l2_values, bins=15, color=VIVID_PURPLE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 1].axvline(l2_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l2_values.mean():.2f}")
axes[0, 1].legend(frameon=False, fontsize=10)

# L∞ distribution
axes[1, 0].set_title(r"$L_\infty$ Distortion Distribution", fontsize=14)
axes[1, 0].set_xlabel(r"$L_\infty$ Distance")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].hist(linf_values, bins=15, color=NUGGET_YELLOW, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 0].axvline(linf_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {linf_values.mean():.2f}")
axes[1, 0].legend(frameon=False, fontsize=10)

# Elastic-net distribution
axes[1, 1].set_title("Elastic-Net Distortion Distribution", fontsize=14)
axes[1, 1].set_xlabel("Elastic Distance")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].hist(elastic_values, bins=15, color=AQUAMARINE, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 1].axvline(elastic_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {elastic_values.mean():.2f}")
axes[1, 1].legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "ead_distortion_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_distortion_analysis.png")
```

Each subplot captures a different view of perturbation magnitude. The L1 distribution counts total absolute change across all pixels, revealing whether modifications spread across many pixels or concentrate on few. The squared L2 distribution measures the sum of squared pixel changes (energy), balancing magnitude against spatial distribution. The L∞ distribution shows maximum per-pixel change, exposing worst-case perturbation intensity. The Elastic-Net distribution combines L1 and squared L2 via β, providing the composite metric our optimization actually minimized.

Narrow distributions with low standard deviation indicate consistent attack behavior. All examples required similar perturbation magnitudes. Wide distributions suggest heterogeneous robustness where some examples flip with minimal perturbations while others require substantial modifications. The red dashed line marks the mean, showing whether the distribution centers symmetrically or skews toward higher distortions.

![Four histograms of L1, squared L2, L∞ and Elastic‑Net distortion with the mean marked](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/ead_distortion_analysis.png)

The histograms show that squared L2 values cluster below 5 (mean ≈4.18) with a long right tail reaching about 21, indicating several harder examples that require larger overall distortion. L1 magnitudes center near 20 (mean ≈20.20) with moderate spread. For L∞, the distribution centers around ∼0.45 and extends up to ∼0.9, so a non-trivial subset of pixels experiences large per‑pixel changes. The Elastic‑Net histogram follows squared L2 closely (mean 4.38 vs 4.18), consistent with β=0.01 placing more weight on the squared L2 term than on L1. Increasing β to 0.05 or 0.1 would shift the Elastic‑Net distribution left as the optimizer prioritizes sparsity over raw distortion.

