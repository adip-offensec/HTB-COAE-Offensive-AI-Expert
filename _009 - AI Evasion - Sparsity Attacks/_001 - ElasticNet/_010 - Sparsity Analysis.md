
# Sparsity Analysis

ElasticNet's defining feature is sparse perturbations. But how sparse are they really? Do modifications concentrate on specific pixels or spread randomly? What explains variation across examples? Heatmaps expose spatial structure that scalar statistics completely miss.

This section analyzes sparsity from two perspectives. First, we examine the relationship between L1 and L2 distances, revealing how sparse perturbations differ from dense ones. Second, we visualize where perturbations actually occur, exposing patterns in spatial concentration. Together, these analyses demonstrate ElasticNet’s ability to generate interpretable, concentrated adversarial perturbations.

## Mixed-Norm Relationship Analysis

How do L1 and L2 distances relate for sparse perturbations? Theory suggests L1≈L2⋅k where k represents the number of modified pixels. Sparse perturbations concentrate energy on few pixels, creating different L1/L2 relationships than dense perturbations. Plotting these distances reveals whether our attack follows expected sparsity patterns or exhibits unexpected behavior.

We also need to understand the sparsity-distortion tradeoff. Does higher sparsity come at the cost of larger L2 distortion? Or can we achieve both simultaneously? This relationship determines whether sparse perturbations represent a true advantage or merely shift the attack-distortion curve:

Code: python

```python
# Visualization 3: Mixed-norm relationship and sparsity-distortion tradeoff
print("Creating mixed-norm analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Color by success (green=success, red=failed)
colors = [HTB_GREEN if s else MALWARE_RED for s in final_success.cpu().numpy()]

# Plot 1: L1 vs L2 relationship
axes[0].set_title(r"$L_1$ vs Squared $L_2$ Distortion Relationship", fontsize=14)
axes[0].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[0].set_ylabel(r"$L_1$ Distance", fontsize=12)
axes[0].scatter(l2_values, l1_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add reference line showing L1 = sqrt(L2) relationship
l2_range = np.linspace(l2_values.min(), l2_values.max(), 100)
axes[0].plot(l2_range, np.sqrt(l2_range) * 8, color=AZURE, linestyle="--",
             linewidth=2, alpha=0.5, label=r"Reference: $L_1 \propto \sqrt{L_2}$")
axes[0].legend(frameon=False, fontsize=10)
axes[0].grid(True, alpha=0.3, color=HACKER_GREY)

# Add success legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=HTB_GREEN, edgecolor=NODE_BLACK, label=f'Success ({final_success.sum()}/{batch_size})'),
    Patch(facecolor=MALWARE_RED, edgecolor=NODE_BLACK, label=f'Failed ({(~final_success).sum()}/{batch_size})')
]
axes[0].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

# Plot 2: Sparsity vs L2 distortion
axes[1].set_title("Sparsity vs Distortion Tradeoff", fontsize=14)
axes[1].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[1].set_ylabel("Sparsity (%)", fontsize=12)
axes[1].scatter(l2_values, sparsity_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add mean lines
axes[1].axvline(l2_values.mean(), color=VIVID_PURPLE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean $L_2$: {l2_values.mean():.2f}")
axes[1].axhline(sparsity_values.mean(), color=AQUAMARINE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean Sparsity: {sparsity_values.mean():.1f}%")
axes[1].legend(frameon=False, fontsize=10)
axes[1].grid(True, alpha=0.3, color=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "ead_success_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_success_analysis.png")
```

The left scatter plot reveals how L1 and squared L2 relate for our sparse perturbations. Points above the reference curve indicate sparser perturbations where energy concentrates on fewer pixels, increasing L1 relative to L2. Points below suggest denser modifications. Green markers show successful attacks, red markers show failures. If failures cluster differently than successes, this indicates certain perturbation patterns transfer better to misclassification.

The right plot exposes the sparsity-distortion tradeoff. Horizontal spread shows L2 variation, vertical spread shows sparsity variation. Clustering in the upper-left corner (high sparsity, low distortion) represents ideal attacks that modify few pixels with small magnitudes. Spread toward lower-right (low sparsity, high distortion) indicates expensive attacks requiring dense modifications. The mean lines partition the space, showing whether most examples achieve above-average sparsity or below-average distortion.

![Two scatter plots: L1 vs squared L2 with a reference curve, and sparsity (%) vs squared L2 with mean lines for 20 attacks](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/ead_success_analysis.png)

Both plots show exclusively green markers, confirming 100% success across 20 examples. On the left, the L1 versus squared L2 relationship sits above the reference curve as expected for sparse perturbations. Most points lie below squared L2≈5 (mean 4.18), with a few outliers up to about 21; L1 ranges from the low teens to the mid‑60s. This spread indicates varying robustness across inputs while preserving the same sparse‑modification pattern.

The right plot shows a moderate sparsity–distortion tradeoff. Most points sit around 45–60% sparsity (mean 52.6%) while squared L2 typically falls between 0 and 5, with several higher‑distortion outliers to the right. The mean lines mark this center of mass, and all points correspond to successful attacks. Higher squared L2 generally coincides with lower sparsity, reflecting denser modifications required by harder inputs.

To interpret points in the scatter, link L1, squared L2, and the number of modified pixels. If perturbations have similar magnitude a across k pixels, then L1=ka and squared L2=ka2, so k≈L12/(squaredL2). For example, L1=60 and squared L2=9 gives k≈602/9=400. On a 784-pixel image, that implies roughly 784−400=384 unchanged pixels, about 49% sparsity. Real perturbations vary in magnitude, so this estimate is an upper bound on k under equal-magnitude assumptions; actual k is often smaller (higher sparsity) when a few pixels carry most of the change.

## Sparsity Pattern Visualization

Sparsity percentages quantify how many pixels change. Heatmaps reveal where those changes occur. Do perturbations concentrate on edges where gradients are large? Do they target specific digit features like loops and intersections? Or do they scatter randomly across the image? Spatial patterns provide insights that scalar statistics miss entirely.

To expose these patterns, we'll visualize perturbation magnitudes for individual examples and aggregate sparsity statistics across the full batch:

Code: python

```python
# Visualization 4: Sparsity analysis
print("Creating sparsity analysis...")

fig = plt.figure(figsize=(16, 8))

# Create grid: 2 rows × 5 columns for examples, 1 row for statistics
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)

# Show perturbation heatmaps for first 10 examples
num_display_sparse = min(10, batch_size)

for i in range(num_display_sparse):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])

    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()

    # Show perturbation with colorbar
    im = ax.imshow(torch.abs(pert), cmap="hot", vmin=0, vmax=pert.abs().max())
    ax.axis("off")
    ax.set_title(f"Example {i+1}", fontsize=10)

# Compute sparsity statistics
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Statistics subplot
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.set_title("Sparsity Distribution Across Examples", fontsize=14)
ax_stats.set_xlabel("Example Index")
ax_stats.set_ylabel("Sparsity (%)")
ax_stats.bar(range(len(sparsity_values)), sparsity_values, color=AQUAMARINE,
             alpha=0.7, edgecolor=NODE_BLACK)
ax_stats.axhline(sparsity_values.mean(), color=MALWARE_RED, linestyle="--",
                 linewidth=2, label=f"Mean: {sparsity_values.mean():.2f}%")
ax_stats.legend(frameon=False, fontsize=10)
ax_stats.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(output_dir / "ead_sparsity_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_sparsity_analysis.png")
```

Each heatmap shows absolute perturbation magnitude for one example, revealing spatial structure. Bright regions indicate large pixel modifications, dark regions indicate minimal changes. The `hot` colormap progresses from black (zero change) through red and orange (moderate change) to white (maximum change), making perturbation concentration immediately visible. If modifications cluster along edges, we see bright boundaries. If they target specific features, we see bright spots at loops or intersections.

The bar chart aggregates sparsity across all examples. Each bar represents one example's sparsity percentage (fraction of pixels unchanged). Height variation reveals robustness heterogeneity: uniform bars suggest consistent attack difficulty, while wide variation indicates some examples resist much more strongly than others. The horizontal red line marks mean sparsity, partitioning examples into above-average and below-average groups.

![Heatmaps of per‑pixel perturbation magnitude for 10 examples and a bar chart of sparsity (%) with a mean line](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/ead_sparsity_analysis.png)

The heatmaps expose clear spatial patterns. Perturbations concentrate heavily along digit boundaries and stroke edges. Example 1 shows bright spots clustered in the lower-right where the digit curves. Example 3 reveals modifications targeting the central loop region. Example 5 displays edge modifications running along the digit's vertical stroke. The pattern is consistent: sparse perturbations attack edge pixels where gradient magnitudes are largest, not scattered pixels chosen arbitrarily.

Why edges? Gradient-based optimization naturally finds the steepest descent directions. For digit classification, edges carry maximum discriminative information. A convolutional neural network's early layers extract edge features, making edge pixels disproportionately influential. Modifying edges efficiently shifts the model's internal representations. Modifying background pixels contributes little because the model has learned to ignore uniform regions.

The bar chart shows mean sparsity near 52.6%, with most examples between roughly 35% and 75%. Harder examples sit toward the lower end of this range, while the most favorable examples approach the mid‑70s. This variation demonstrates how digit complexity and model confidence interact: some inputs flip with relatively few unchanged pixels, whereas tougher inputs require denser modifications to succeed.

