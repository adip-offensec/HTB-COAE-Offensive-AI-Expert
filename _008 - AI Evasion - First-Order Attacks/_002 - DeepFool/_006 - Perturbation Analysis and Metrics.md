
Having generated batch adversarial examples, we now analyze their characteristics through spatial perturbation patterns and statistics. Where does DeepFool concentrate modifications? How do perturbation magnitudes distribute across samples? Which digit classes prove most vulnerable? These visualizations and statistics expose the geometric properties of minimal adversarial perturbations.

## Spatial Perturbation Analysis

Theory predicts DeepFool will concentrate modifications along decision boundaries where class-discriminative features reside. Does empirical evidence support this? We need spatial heatmaps showing where DeepFool actually modifies pixels across multiple samples. The challenge: raw perturbation magnitudes are minimal (often `[-0.5, 0.5]`), making patterns invisible without amplification. We'll create two visualization layers: raw perturbation heatmaps using diverging colormaps (red for positive changes, blue for negative), and amplified difference visualizations (10x magnification) overlaid to expose subtle modifications. This dual approach validates whether DeepFool targets semantically meaningful regions or scatters noise uniformly.

Code: python

```python
def visualize_perturbation_analysis(results, save_dir='output'):
    """
    Analyze and visualize perturbation characteristics across samples.

    Creates two-row visualization: top shows raw perturbation heatmaps,
    bottom shows amplified differences overlaid on originals.

    Args:
        results (list): Attack results
        save_dir (str): Output directory
    """
    print("\nGenerating perturbation analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    # Select first 3 successful attacks
    successful_attacks = [r for r in results if r['success']][:3]

    for idx, result in enumerate(successful_attacks):
        # Top row: Raw perturbation heatmap
        ax_top = axes[0, idx]
        pert = result['perturbation'].squeeze().numpy()
        vmax = np.abs(pert).max() or 1e-6
        im_top = ax_top.imshow(pert, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_top.set_title(f'Perturbation (L2={result["l2_norm"]:.3f})',
                        color=HTB_GREEN, fontsize=10)
        ax_top.axis('off')

        cbar_top = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.04)
        cbar_top.outline.set_edgecolor(HACKER_GREY)
        cbar_top.ax.tick_params(colors=WHITE)

        # Bottom row: Amplified difference visualization
        ax_bottom = axes[1, idx]
        orig_img = result['original_image'].squeeze().numpy()
        adv_img = result['perturbed_image'].squeeze().detach().numpy()
        diff_amplified = (adv_img - orig_img) * 10  # 10x amplification for visibility

        im_bottom = ax_bottom.imshow(diff_amplified, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_bottom.set_title(f"{result['original_label']} → {result['adversarial_label']} "
                           f"({result['iterations']} iters)",
                           color=NUGGET_YELLOW, fontsize=10)
        ax_bottom.axis('off')

        cbar_bottom = plt.colorbar(im_bottom, ax=ax_bottom, fraction=0.046, pad=0.04)
        cbar_bottom.outline.set_edgecolor(HACKER_GREY)
        cbar_bottom.ax.tick_params(colors=WHITE)

    plt.suptitle('DeepFool Perturbation Analysis', color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_perturbations.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Perturbation analysis saved to {save_dir}/deepfool_perturbations.png")

# Generate the perturbation analysis visualization
visualize_perturbation_analysis(results, save_dir='output')
```

![Two-row DeepFool perturbation analysis showing perturbation heatmaps with L2 norms above and amplified difference maps such as 7→2 and 1→8 below.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/deepfool_perturbations.png)

The `RdBu_r` colormap (reversed Red-Blue) creates intuitive diverging visualization: red shows positive perturbations (brightening pixels), blue shows negative perturbations (darkening), and white indicates no change. The symmetric value range `vmin=-vmax, vmax=vmax` centers white at zero, ensuring neutral pixels remain uncolored. The amplified difference calculation `(adv_img - orig_img) * 10` magnifies subtle modifications 10x: a 0.03 pixel change becomes 0.3, visually detectable after colormap application. Without amplification, changes of 0.01-0.05 appear uniform grey.

The heatmaps show concentrated modifications along digit boundaries and distinctive features. The visualization shows three successful attacks, with the first being our ’7’ flipped to ’2’ attack (L2=7.72). For this attack, perturbations concentrate along the digit stroke, with modifications targeting the areas where ’7’ structure must transform into ’2’ shape. Background regions remain white (unchanged), confirming DeepFool targets semantically meaningful areas where small changes maximally impact classification, not uniform noise scattering. The second and third panels show ’2→6’ (L2=4.10) and ’1→8’ (L2=5.16) transitions, each displaying concentrated modifications at class-discriminative boundaries.

## Statistical Distribution Analysis

Spatial heatmaps show local targeting, but population-level statistics quantify global patterns. What’s the typical perturbation magnitude across samples? How many iterations does DeepFool usually require? Which digit classes prove most vulnerable? We need three statistical visualizations: L2 norm distribution (showing perturbation magnitude clustering), iteration count distribution (measuring convergence efficiency), and per-class success rates (identifying vulnerable digits). These metrics enable comparing DeepFool’s behavior across models, datasets, or attack configurations.

Code: python

```python
print("\nGenerating attack metrics visualization...")

# Setup three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)
    ax.tick_params(colors=WHITE)
    ax.grid(True, alpha=0.3, color=HACKER_GREY, linestyle='--')

# Panel 1: L2 Norm Distribution
l2_norms = [r['l2_norm'] for r in results]
axes[0].hist(l2_norms, bins=15, color=HTB_GREEN, alpha=0.7, edgecolor=HACKER_GREY)
axes[0].set_xlabel('L2 Norm', color=WHITE)
axes[0].set_ylabel('Frequency', color=WHITE)
axes[0].set_title('Perturbation Magnitude Distribution', color=HTB_GREEN)
print(f"L2 norm range: [{min(l2_norms):.4f}, {max(l2_norms):.4f}]")

# Panel 2: Iteration Count Distribution
iterations = [r['iterations'] for r in results]
axes[1].hist(iterations, bins=range(1, max(iterations)+2),
            color=AZURE, alpha=0.7, edgecolor=HACKER_GREY)
axes[1].set_xlabel('Iterations', color=WHITE)
axes[1].set_ylabel('Frequency', color=WHITE)
axes[1].set_title('Iterations Required', color=HTB_GREEN)
print(f"Iteration range: [{min(iterations)}, {max(iterations)}]")

# Panel 3: Per-Class Success Rates
class_success = {}
for r in results:
    orig = r['original_label']
    if orig not in class_success:
        class_success[orig] = {'total': 0, 'success': 0}
    class_success[orig]['total'] += 1
    if r['success']:
        class_success[orig]['success'] += 1

classes = sorted(class_success.keys())
success_rates = [
    class_success[c]['success'] / class_success[c]['total'] * 100
    if class_success[c]['total'] > 0 else 0
    for c in classes
]

bars = axes[2].bar(classes, success_rates, color=NUGGET_YELLOW,
                   alpha=0.7, edgecolor=HACKER_GREY)
axes[2].set_xlabel('Original Class', color=WHITE)
axes[2].set_ylabel('Success Rate (%)', color=WHITE)
axes[2].set_title('Attack Success by Class', color=HTB_GREEN)
axes[2].set_ylim(0, 105)

# Add percentage labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax_x = bar.get_x() + bar.get_width() / 2.0
    axes[2].text(ax_x, height + 1, f'{rate:.0f}%',
                 ha='center', va='bottom', color=WHITE, fontsize=8)

# Save visualization
plt.suptitle('DeepFool Attack Metrics', color=HTB_GREEN, fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('output/deepfool_metrics.png',
            facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
plt.close()

print("Metrics visualization saved to output/deepfool_metrics.png")
```

![Three-panel DeepFool metrics view with L2 norm histogram, iteration histogram, and per-class bar chart showing near-100% success for each digit.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/deepfool_metrics.png)

The L2 norm histogram with 15 bins shows distribution shape across the range [0.60, 7.72]. Most perturbations cluster around the 4-5 average, with the minimum at 0.60 representing the easiest boundary crossing (digit ’3→5’, structurally similar) and maximum at 7.72 representing the most challenging transformation (digit ’7→2’, requiring substantial structural changes). The `alpha=0.7` transparency creates softer visual appearance while maintaining distinct bar boundaries via `edgecolor`. The iteration histogram uses integer bins `range(1, max(iterations)+2)` ensuring each count gets its own bar. The `+2` compensates for range’s exclusive endpoint. The distribution shows iterations from 1 to 5, with most samples converging in 2-4 iterations (averaging exactly 3.0), demonstrating efficient convergence across diverse digit pairs.

The per-class success computation uses a two-counter dictionary: `total` tracks samples per class, `success` counts classification flips. The percentage formula `success / total * 100` converts to success rate, with zero-division protection for classes with no samples. The bar chart text labels use `bar.get_x() + bar.get_width() / 2.0` for horizontal centering and `height + 1` for vertical positioning 1% above bar tops. The `ylim(0, 105)` provides 5% headroom preventing label clipping. Per-class rates indicate which digits (e.g., '1' vs '8') prove more vulnerable to minimal perturbations based on decision boundary geometry.

## Aggregate Attack Summary

Visualizations show patterns, but summary statistics provide quantitative benchmarks for comparing DeepFool across different models or datasets. We need consolidated metrics answering: what's the overall success rate, what's the average perturbation magnitude, what's typical convergence speed, and which class transitions occur most frequently? These aggregate statistics enable reproducible comparisons and identify systematic misclassification patterns (e.g., '7→9' transitions might occur more often than '0→1' if those digit pairs have closer decision boundaries).

Code: python

```python
def print_summary_statistics(results):
    """
    Print summary statistics for attack results.

    Computes and displays success rate, perturbation statistics, iteration
    statistics, and common class transitions.

    Args:
        results (list): Attack results
    """
    print("\n" + "="*60)
    print("Attack Summary Statistics")
    print("="*60)

    successful_attacks = [r for r in results if r['success']]

    if successful_attacks:
        avg_l2 = np.mean([r['l2_norm'] for r in successful_attacks])
        avg_iterations = np.mean([r['iterations'] for r in successful_attacks])
        min_l2 = min([r['l2_norm'] for r in successful_attacks])
        max_l2 = max([r['l2_norm'] for r in successful_attacks])

        print(f"Success Rate: {len(successful_attacks)}/{len(results)} "
              f"({100*len(successful_attacks)/len(results):.1f}%)")
        print(f"Average L2 Norm: {avg_l2:.4f}")
        print(f"L2 Range: [{min_l2:.4f}, {max_l2:.4f}]")
        print(f"Average Iterations: {avg_iterations:.1f}")

        # Class transition analysis
        transitions = {}
        for r in successful_attacks:
            key = f"{r['original_label']}→{r['adversarial_label']}"
            transitions[key] = transitions.get(key, 0) + 1

        print(f"\nMost Common Misclassifications:")
        for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {trans}: {count} times")
    else:
        print("No successful attacks generated")

    print("="*60)

# Generate summary
print_summary_statistics(results)
```

Expected output:

Code: txt

```txt
============================================================
Attack Summary Statistics
============================================================
Success Rate: 20/20 (100.0%)
Average L2 Norm: 4.8661
L2 Range: [0.5960, 7.7214]
Average Iterations: 3.0

Most Common Misclassifications:
  9→4: 3 times
  7→2: 1 times
  2→6: 1 times
  1→8: 1 times
  0→6: 1 times
============================================================
```

The transition frequency analysis uses dictionary accumulation: `transitions[key] = transitions.get(key, 0) + 1` increments counts for each `original→adversarial` pairing. The sorting `sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]` ranks by frequency (second tuple element `x[1]`) in descending order, selecting top 5. The most common transition '9→4' appears 3 times, indicating that these digit pairs have a particularly accessible decision boundary in the learned feature space. This makes geometric sense: both '9' and '4' have similar upper loops and vertical strokes, requiring smaller perturbations to bridge the gap compared to very different shapes like '0→1'.

