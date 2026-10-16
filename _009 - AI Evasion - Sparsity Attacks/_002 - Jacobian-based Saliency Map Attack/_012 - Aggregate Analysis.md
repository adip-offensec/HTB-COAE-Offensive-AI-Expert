
# Aggregate Analysis

Beyond individual attack visualizations, analyzing results across multiple attacks reveals patterns in sparsity, success rates, and attack efficiency. This section implements visualizations for L0 distributions and side-by-side comparisons of single-pixel versus pairwise approaches across the full test set.

## L0 Distribution Analysis

Analyzing L0 norms across multiple attacks shows typical sparsity levels. We'll break this into setup, plotting, and statistics phases for clarity.

### Creating L0 Histogram

Displaying the distribution reveals sparsity patterns across the attack dataset:

Code: python

```python
def visualize_l0_distribution(l0_values, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, NUGGET_YELLOW, AZURE

    # Create figure and histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(l0_values, bins=30, color=HTB_GREEN, alpha=0.7)

    # Compute statistics
    mean_l0 = np.mean(l0_values)
    median_l0 = np.median(l0_values)
    min_l0 = np.min(l0_values)
    max_l0 = np.max(l0_values)
```

Setting 30 bins provides sufficient granularity for L0 values typically ranging from 10-120 pixels without over-segmenting the distribution. Alpha transparency at 0.7 allows gridlines to show through, maintaining readability. Computing all statistics upfront enables consistent use across plotting and console output.

### Overlaying Statistical Markers

Vertical lines highlight central tendency measures:

Code: python

```python
    # Add statistical markers
    ax.axvline(mean_l0, color=NUGGET_YELLOW, linestyle='--', linewidth=2,
               label=f'Mean: {mean_l0:.1f}')
    ax.axvline(median_l0, color=AZURE, linestyle='--', linewidth=2,
               label=f'Median: {median_l0:.1f}')

    # Formatting
    ax.set_xlabel('L0 Norm (Pixels Modified)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('L0 Norm Distribution', color=HTB_GREEN,
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
```

Dashed vertical lines use contrasting colors from the HTB secondary palette: yellow for mean, blue for median. When mean exceeds median (right-skewed distribution), this reveals that occasional high-L0 attacks pull the average up. The grid with alpha=0.3 provides reference without cluttering. Legend placement defaults to optimal location based on data density.

### Saving and Reporting

Finalizing the figure and printing comprehensive statistics:

Code: python

```python
    # Save
    plt.tight_layout()
    save_path = output_dir / 'jsma_l0_distribution.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight')
    plt.close()

    # Report statistics
    print(f"L0 distribution saved to {save_path}")
    print(f"  Mean L0: {mean_l0:.1f}")
    print(f"  Median L0: {median_l0:.1f}")
    print(f"  Min L0: {min_l0:.0f}")
    print(f"  Max L0: {max_l0:.0f}")
```

Console output provides numerical summary complementing the visual distribution. Formatting to one decimal place for mean and median balances precision with readability. Integer formatting for min and max reflects that L0 norms are discrete counts. This dual reporting (visual + numerical) supports both quick interpretation and detailed analysis.

Execute this function with L0 values from batch attacks:

Code: python

```python
# Combine single-pixel and pairwise L0 values
all_l0_values = results['pixels_modified'] + results_pairs['pixels_modified']

visualize_l0_distribution(all_l0_values, output_dir)
```

Output:

Code: txt

```txt
L0 distribution saved to output/jsma_l0_distribution.png
  Mean L0: 58.2
  Median L0: 55.0
  Min L0: 12
  Max L0: 118
```

![Histogram of L0 (pixels modified) with dashed mean and median markers](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jsma_l0_distribution.png)

This histogram shows the typical sparsity achieved by JSMA across many attacks, revealing how many pixels are typically needed for successful misclassification.

## Attack Comparison Visualization

Comparing single-pixel and pairwise attacks side-by-side across all tested samples reveals their relative effectiveness. We create a comprehensive grid showing original images, perturbations, and results for both attack variants. To keep functions manageable, we'll break this into composable helpers.

### Data Preparation Helper

Converting tensors and computing perturbations for all samples provides the data foundation:

Code: python

```python
def prepare_comparison_data(original_images, single_adversarial, pairwise_adversarial):
    n_samples = len(original_images)
    data = []

    for idx in range(n_samples):
        orig = original_images[idx].cpu().numpy().squeeze()
        single_adv = single_adversarial[idx].cpu().numpy().squeeze()
        pair_adv = pairwise_adversarial[idx].cpu().numpy().squeeze()

        single_pert = np.abs(single_adv - orig)
        pair_pert = np.abs(pair_adv - orig)

        single_pert_norm = single_pert / (single_pert.max() + 1e-8)
        pair_pert_norm = pair_pert / (pair_pert.max() + 1e-8)

        data.append({
            'orig': orig,
            'single_adv': single_adv,
            'pair_adv': pair_adv,
            'single_pert_norm': single_pert_norm,
            'pair_pert_norm': pair_pert_norm,
            'l0_single': int((single_pert > 0).sum()),
            'l0_pair': int((pair_pert > 0).sum())
        })

    return data
```

Each sample gets converted once, with all derived data (perturbations, L0 norms) computed upfront. Normalizing perturbations by their maximum ensures visibility across varying magnitude ranges. The dictionary structure organizes related data, avoiding parallel arrays that would complicate indexing.

### Original Images Row Helper

Plotting the original images establishes the baseline for visual comparison:

Code: python

```python
def plot_original_row(axes, data, original_labels, HTB_GREEN):
    for idx, sample_data in enumerate(data):
        axes[0, idx].imshow(sample_data['orig'], cmap='gray', interpolation='nearest')
        if idx == 0:
            axes[0, idx].set_ylabel('Original', color=HTB_GREEN, fontsize=11, fontweight='bold')
        axes[0, idx].text(0.02, 0.98, f'#{idx+1}\nTrue: {original_labels[idx]}',
                         transform=axes[0, idx].transAxes,
                         verticalalignment='top',
                         fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[0, idx].axis('off')
```

Row labels appear only on the first column to avoid clutter. Text annotations use axes transform coordinates (0-1 range independent of data units), positioning consistently across samples. The bounding box with alpha=0.7 provides contrast against varying image backgrounds.

### Perturbation Row Helper

Displaying perturbation heatmaps requires computing L0 norms and color-coding by magnitude:

Code: python

```python
def plot_perturbation_row(axes, row, data, label, HTB_GREEN):
    key = 'single_pert_norm' if 'Single' in label else 'pair_pert_norm'
    l0_key = 'l0_single' if 'Single' in label else 'l0_pair'

    for idx, sample_data in enumerate(data):
        axes[row, idx].imshow(sample_data[key], cmap='hot', interpolation='nearest')
        if idx == 0:
            axes[row, idx].set_ylabel(label, color=HTB_GREEN, fontsize=11, fontweight='bold')
        axes[row, idx].text(0.5, 0.02, f'L0={sample_data[l0_key]}',
                           transform=axes[row, idx].transAxes,
                           horizontalalignment='center',
                           fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, idx].axis('off')
```

Generic key selection based on label text avoids duplicating code for single-pixel vs pairwise rows. Centering L0 annotations at the bottom (0.5 horizontal, 0.02 vertical) reserves the top for other information if needed. The hot colormap highlights high-magnitude perturbations in yellow/red against dark backgrounds.

### Result Row Helper

Displaying adversarial results requires color-coding by success/failure:

Code: python

```python
def plot_result_row(axes, row, data, label, predictions, targets, HTB_GREEN, MALWARE_RED):
    key = 'single_adv' if 'Single' in label else 'pair_adv'

    for idx, sample_data in enumerate(data):
        axes[row, idx].imshow(sample_data[key], cmap='gray', interpolation='nearest')
        if idx == 0:
            axes[row, idx].set_ylabel(label, color=HTB_GREEN, fontsize=11, fontweight='bold')
        success = (predictions[idx] == targets[idx])
        result_color = HTB_GREEN if success else MALWARE_RED
        axes[row, idx].text(0.5, 0.02, f'Pred: {predictions[idx]}',
                           transform=axes[row, idx].transAxes,
                           horizontalalignment='center',
                           fontsize=8,
                           color=result_color,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, idx].axis('off')
```

Success determination compares predicted class against target class, coloring predictions green for matches (successful attacks) and red for mismatches (failures). This visual encoding enables immediate assessment of attack effectiveness across the sample set.

### Main Comparison Function

Now we assemble the helpers into the complete visualization:

Code: python

```python
def visualize_attack_comparison(original_images, single_adversarial, pairwise_adversarial,
                                original_labels, predicted_single, predicted_pairs,
                                target_labels, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, MALWARE_RED

    n_samples = len(original_images)

    # Prepare data
    data = prepare_comparison_data(original_images, single_adversarial, pairwise_adversarial)

    # Create grid
    fig, axes = plt.subplots(5, n_samples, figsize=(2.5 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    # Plot rows
    plot_original_row(axes, data, original_labels, HTB_GREEN)
    plot_perturbation_row(axes, 1, data, 'Single-Pixel\nPerturbation', HTB_GREEN)
    plot_result_row(axes, 2, data, 'Single-Pixel\nResult', predicted_single, target_labels, HTB_GREEN, MALWARE_RED)
    plot_perturbation_row(axes, 3, data, 'Pairwise\nPerturbation', HTB_GREEN)
    plot_result_row(axes, 4, data, 'Pairwise\nResult', predicted_pairs, target_labels, HTB_GREEN, MALWARE_RED)

    # Save
    plt.tight_layout()
    save_path = output_dir / 'jsma_attack_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Attack comparison visualization saved to {save_path}")
```

Our main function coordinates workflow by preparing data, creating the figure grid, plotting each row using appropriate helpers, finalizing layout, and saving. This composition pattern keeps each function focused on a single responsibility while maintaining clear data flow. Reshape handling ensures consistent axes indexing whether plotting 1 or 10 samples.

Execute this function to generate the comparison grid:

Code: python

```python
# Get predictions for both attack variants
predicted_single = []
predicted_pairs = []

with torch.no_grad():
    for i in range(len(original_images)):
        pred_s = model(results['adversarial'][i]).argmax(dim=1).item()
        pred_p = model(results_pairs['adversarial'][i]).argmax(dim=1).item()
        predicted_single.append(pred_s)
        predicted_pairs.append(pred_p)

visualize_attack_comparison(
    original_images,
    results['adversarial'],
    results_pairs['adversarial'],
    original_labels,
    predicted_single,
    predicted_pairs,
    target_labels,
    output_dir
)
```

Output:

Code: txt

```txt
Attack comparison visualization saved to output/jsma_attack_comparison.png
```

![Grid comparing single‑pixel and pairwise JSMA across 10 samples: originals, perturbations, and results](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jsma_attack_comparison.png)

Looking at the comparison grid visualization (if generated with actual test results), we would see stark differences between single-pixel and pairwise JSMA across 10 test samples. The top row would show the original digits (7, 2, 1, 0, 4, 1, 4, 9, 5, 9) with their true labels. Each subsequent row pair would contrast single-pixel and pairwise results.

Row 2 (single-pixel perturbations) would show modifications with L0 counts ranging from 60 to 100 pixels. Most samples exhausted their iteration budgets trying to find effective modifications. The perturbations appear as scattered hot spots (yellow/red) against the black background, but with `theta=0.25`, the individual pixel changes are too subtle to cross decision boundaries.

Row 3 (single-pixel results) would reveal universal failure: all 10 samples show red predictions (failures), confirming that none achieved their target misclassification despite modifying an average of 93.9 pixels each. The model's predictions remain at their original classes for all samples.

Row 4 (pairwise perturbations) shows a dramatically different pattern. Most samples have sparse modifications (L0 ranging from 12 to 68 pixels), but sample 2 stands out with L0=118, indicating it required extensive modifications even with pairwise selection and ultimately failed. Sample 4 achieves remarkable sparsity with only 12 pixels modified. The spatial distribution differs from single-pixel, as pairwise saliency discovers different synergistic combinations, and `theta=1.0` enables full saturation per modification.

Row 5 (pairwise results) demonstrates the transformation: 9 out of 10 samples show green predictions (success), with only sample 2 displaying a red prediction (failure). Every sample that single-pixel failed to attack, pairwise succeeds (except sample 2). Sample 4's success with only 12 pixels exemplifies pairwise JSMA's ability to find highly effective sparse perturbations through feature synergies.

This visual comparison makes the quantitative results concrete: pairwise JSMA's 90% success rate versus single-pixel's 0% (with `theta=0.25`) represents a transformative improvement. We see consistent success in finding adversarial examples where the single-pixel baseline completely fails, achieving success with dramatically fewer pixel modifications (12-68 pixels for successful pairwise attacks vs 60-100 pixels for failed single-pixel attempts).

Our visualization functions integrate seamlessly with the attack code, making it easy to generate diagnostic plots after running attacks. This tight integration between attack execution and analysis helps identify why attacks succeed or fail, revealing patterns in saliency maps, perturbation distributions, and convergence behavior.

