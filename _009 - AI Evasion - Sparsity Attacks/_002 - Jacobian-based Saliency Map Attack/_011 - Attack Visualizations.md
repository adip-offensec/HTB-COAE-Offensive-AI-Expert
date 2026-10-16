
# Attack Visualizations

Visualization reveals JSMA's behavior across multiple dimensions: the attack transformation process, progress tracking over iterations, and perturbation structure analysis. Each visualization function consumes attack results and generates matplotlib figures adhering to the HTB visual theme.

## Attack Process Visualization

Understanding attack mechanics requires seeing the full transformation pipeline. We'll build a 4-panel figure capturing the original image, adversarial result, perturbation magnitude heatmap, and binary modification mask. Breaking this into helpers maintains manageable function sizes.

### Data Preparation for Attack Process

Converting inputs and computing derived metrics provides the foundation for visualization:

Code: python

```python
def prepare_attack_process_data(original, adversarial):
    # Convert to numpy
    if isinstance(original, torch.Tensor):
        orig = original.cpu().numpy()
    else:
        orig = original
    if isinstance(adversarial, torch.Tensor):
        adv = adversarial.cpu().numpy()
    else:
        adv = adversarial

    # Squeeze and compute perturbations
    orig = np.squeeze(orig)
    adv = np.squeeze(adv)
    pert = adv - orig
    pert_mag = np.abs(pert)
    mod_map = (pert_mag > 0)

    # Compute norms
    l0 = int(mod_map.sum())
    l2 = float(np.linalg.norm(pert))
    linf = float(pert_mag.max() if pert_mag.size else 0.0)

    return {
        'orig': orig,
        'adv': adv,
        'pert_mag': pert_mag,
        'mod_map': mod_map,
        'l0': l0,
        'l2': l2,
        'linf': linf
    }
```

Type checking with `isinstance` handles both tensor and numpy array inputs gracefully. Squeezing removes singleton dimensions, converting $(1, 1, 28, 28)$ to $(28, 28)$ for plotting. Computing perturbation magnitudes requires taking absolute values since we care about change magnitude regardless of direction. The modification map uses a boolean comparison to identify any nonzero perturbation. Computing all three norms upfront avoids recalculating during plotting.

### Main Attack Process Visualization

With data prepared, we create the 4-panel figure:

Code: python

```python
def visualize_attack_process(original, adversarial, original_class, target_class,
                             predicted_class, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, MALWARE_RED

    # Prepare data
    data = prepare_attack_process_data(original, adversarial)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    success = (predicted_class == target_class)

    # Original
    axes[0].imshow(data['orig'], cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Original\nTrue: {original_class}',
                      color=HTB_GREEN, fontsize=12)
    axes[0].axis('off')

    # Adversarial
    axes[1].imshow(data['adv'], cmap='gray', interpolation='nearest')
    axes[1].set_title(f'Adversarial\nPred: {predicted_class}  Target: {target_class}  ' +
                      ('Success' if success else 'Fail'),
                      color=HTB_GREEN if success else MALWARE_RED, fontsize=12)
    axes[1].axis('off')

    # Perturbation magnitude
    pm_norm = data['pert_mag'] / (data['pert_mag'].max() + 1e-8)
    axes[2].imshow(pm_norm, cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Perturbation |Δ|\nL0={data["l0"]}  L2={data["l2"]:.3f}  L∞={data["linf"]:.3f}',
                      color=HTB_GREEN, fontsize=12)
    axes[2].axis('off')

    # Modified pixels
    axes[3].imshow(data['mod_map'], cmap='hot', interpolation='nearest')
    total = int(data['orig'].size)
    percent = 100.0 * data['l0'] / max(1, total)
    axes[3].set_title(f'Modified Pixels\n{data["l0"]}/{total} ({percent:.2f}%)',
                      color=HTB_GREEN, fontsize=12)
    axes[3].axis('off')

    # Save
    plt.tight_layout()
    save_path = output_dir / 'jsma_attack_process.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight')
    plt.close()

    print(f"Attack process visualization saved to {save_path}")
```

We use grayscale colormap for original and adversarial images to maintain digit recognizability. Coloring titles with green for successful attacks and red for failures provides immediate visual feedback. Normalizing perturbation magnitude ensures visibility regardless of actual values (without normalization, small perturbations would appear uniformly dark). Our hot colormap highlights modified regions in yellow/red. Computing the percentage modified provides intuitive sparsity interpretation.

Now let's execute this function with actual attack results:

Code: python

```python
# Execute visualization with first successful attack
with torch.no_grad():
    pred_adv = model(results_pairs['adversarial'][0]).argmax(dim=1).item()

visualize_attack_process(
    original_images[0],
    results_pairs['adversarial'][0],
    original_labels[0],
    target_labels[0],
    pred_adv,
    output_dir
)
```

Output:

Code: txt

```txt
Attack process visualization saved to output/jsma_attack_process.png
```

![Four‑panel JSMA example: original digit, adversarial classification, perturbation heatmap, and modified‑pixel mask](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jsma_attack_process.png)

Each panel contributes a distinct perspective on attack mechanics. We establish baseline appearance with the original digit. Next, the adversarial result reveals whether perturbations achieved misclassification (green title for success, red for failure). The heatmap exposes which spatial regions changed most, with hot colors indicating larger magnitude shifts. Finally, the binary mask quantifies sparsity visually through white pixels on black background, making the L0 count immediately apparent.

### Understanding the Generated Visualization

Examining the attack process visualization for sample 1 (digit 7 → target 2) reveals the pairwise attack's behavior. Looking at the original panel, we see a clean handwritten "7" with the characteristic vertical stroke and angled top. The adversarial panel shows the result after 68 pixels were modified across 35 iterations. The digit now appears distorted, with portions of the top stroke removed and the vertical section modified. The model successfully misclassifies it as "2", achieving the attack's goal.

Notice how the perturbation magnitude heatmap uses a "hot" colormap where yellow and red indicate the strongest changes, while darker regions show unmodified pixels. Modifications concentrate along the top horizontal section and the vertical stroke, representing strategic locations where JSMA determined changes would most effectively shift the classification from 7 to 2. Notice the scattered pattern rather than uniform noise, characteristic of sparse L0-minimizing attacks.

Looking at the rightmost panel, we see the binary mask displaying modified pixels in white against the black background. The sparse nature becomes immediately apparent: only 68 out of 784 pixels changed (8.67%). Examining the norm metrics reveals the attack's character: L0=68 confirms sparsity, L2≈5.5 shows moderate total perturbation magnitude, and L∞=1.0 indicates pixels saturated to their maximum values. This aggressive per-pixel modification (with `theta=1.0`, pixels jump fully from 0.0 to 1.0 or vice versa) allows the attack to succeed with relatively few pixel changes, trading off individual pixel magnitude for overall sparsity.

## Progress Tracking Visualization

Tracking target confidence over iterations reveals convergence behavior:

Code: python

```python
def visualize_attack_progress(stats, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN, NUGGET_YELLOW

    its = stats['iterations']
    conf = stats['target_confidence']
    pix = stats['pixels_modified']

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax2 = ax.twinx()

    ax.plot(its, conf, color=HTB_GREEN, linewidth=2.2, label='Target Confidence')
    ax2.plot(its, pix, color=NUGGET_YELLOW, linewidth=2.0, label='Pixels Modified')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Target Confidence')
    ax2.set_ylabel('Pixels Modified')
    ax.grid(True, alpha=0.35)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_path = output_dir / 'jsma_progress.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight')
    plt.close()

    print(f"Progress visualization saved to {save_path}")
```

Execute this function with attack statistics from a single-pixel attack:

Code: python

```python
# Assume stats collected during single-pixel attack in Single-Pixel Attack
visualize_attack_progress(stats, output_dir)
```

Output:

Code: txt

```txt
Progress visualization saved to output/jsma_progress.png
```

![Dual‑axis line chart: target confidence rises with iterations alongside pixels modified](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jsma_progress.png)

Dual-axis plotting overlays two metrics with different scales on a single timeline. Target confidence (green, left axis) measures attack effectiveness directly: rising from near-zero toward 1.0 signals successful manipulation. Pixels modified (yellow, right axis) counts cumulative L0 norm, showing resource consumption. Steep confidence increases with gradual pixel growth indicate efficient attacks. Plateaued confidence despite growing pixel counts signal diminishing returns or failure modes.

## Saliency Map Visualization

Visualizing the saliency map at key iterations shows which regions JSMA considers most important:

Code: python

```python
def visualize_saliency_map(alpha, beta, search_space, original_shape, output_dir):
    from htb_ai_library.visualization.styles import HTB_GREEN

    # Compute saliency scores
    inc_scores = score_increase_saliency(alpha, beta)
    dec_scores = score_decrease_saliency(alpha, beta)
    combined_scores = np.maximum(inc_scores, dec_scores)

    # Apply search mask
    combined_scores = combined_scores * search_space

    # Reshape to image dimensions
    h, w = original_shape[-2:]
    saliency_map = combined_scores.reshape(h, w)

    # Create visualization
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(saliency_map, cmap='hot', interpolation='nearest')
    ax.set_title('Saliency Map\n(Higher = More Important)',
                 color=HTB_GREEN, fontsize=14, fontweight='bold')
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Saliency Score')

    plt.tight_layout()
    save_path = output_dir / 'jsma_saliency_map.png'
    plt.savefig(save_path, dpi=175, bbox_inches='tight')
    plt.close()

    print(f"Saliency map saved to {save_path}")
```

Execute this function to visualize saliency for the first sample:

Code: python

```python
# Compute gradients for visualization
jacobian = compute_jacobian_matrix(original_images[0], model, 10, 'logits')
alpha = extract_target_gradient(jacobian, target_labels[0])
beta = extract_other_gradients(jacobian, target_labels[0])
search_space = initialize_search_space(original_images[0].shape)

visualize_saliency_map(alpha, beta, search_space, original_images[0].shape, output_dir)
```

Output:

Code: txt

```txt
Saliency map saved to output/jsma_saliency_map.png
```

![Saliency heatmap highlighting pixels most influential for the target class, with colorbar](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jsma_saliency_map.png)

Our saliency map reveals which regions of the image have the highest influence on the target class, often highlighting semantic features like digit strokes or object edges.

The next section implements aggregate visualizations analyzing L0 distributions and comparing single-pixel versus pairwise attack results across the full test set.

