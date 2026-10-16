
# Batch Attack Generation

The single-image demonstration showed DeepFool's ability to find minimal perturbations. One sample. One perturbation. One data point. What patterns emerge when attacking multiple samples? Does perturbation size vary by digit class? Which samples resist attacks most strongly, and which crumble immediately? Batch analysis across diverse inputs exposes statistical patterns completely invisible in single examples, transforming anecdotal observations into quantitative measurements.

## Systematic Sample Processing

Single-sample attacks provide proof of concept, but statistical patterns require population-level analysis. Does DeepFool consistently converge in similar iteration counts? Do all digit classes exhibit comparable robustness? Which decision boundaries require larger perturbations? We need to attack multiple diverse samples while tracking metrics for each: perturbation norms, iteration counts, success indicators, and classification trajectories. The batch processing infrastructure will generate this dataset for statistical analysis.

### Establishing the Batch Attack Pipeline

Our goal is processing many samples efficiently while preserving per-sample metrics. Why not use larger batches? DeepFool's iterative nature makes each sample independent: one may converge in 2 iterations while another requires 6. Batch processing would force all samples to wait for the slowest, wasting computation. We'll process 20 samples individually to balance statistical insight with runtime, accumulating results in a structured format for downstream analysis.

Code: python

```python
num_examples = 20
print(f"\nGenerating {num_examples} adversarial examples using DeepFool...")

_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
model.eval()

results = []
success_count = 0

print(f"Test loader ready with {len(test_loader.dataset)} samples")
print(f"Will process first {num_examples} samples")
print("Starting batch attack generation...")
```

The single-sample batches enable independent processing where each attack terminates immediately upon convergence without waiting for others. The `results` list accumulates dictionaries containing both tensor data (images, perturbations) and scalar metrics (norms, iteration counts). The `success_count` tracks classification flips separately for quick summary statistics. Processing 20 samples takes roughly 1-2 seconds on GPU (50-100ms per sample), sufficient to expose patterns without extensive wait times.

### Executing Attacks and Collecting Metrics

Each sample must be attacked independently, with metrics captured for statistical analysis. We need to store not just success/failure, but the complete trajectory: how many iterations were required, what perturbation magnitude was needed, whether the original prediction was correct, and what the adversarial label became. This rich dataset enables answering questions about convergence patterns, robustness variation across classes, and boundary geometry characteristics. The processing loop must also provide real-time feedback so we can monitor attack effectiveness as samples are processed.

Code: python

```python
for idx, (data, target) in enumerate(test_loader):
    if idx >= num_examples:
        break

    data = data.to(device)

    # Execute DeepFool attack
    r, iterations, orig_label, adv_label, pert_image = deepfool(
        data, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
    )

    # Track success and store metrics
    success = (orig_label != adv_label)
    if success:
        success_count += 1

    results.append({
        'original_image': data.cpu(),
        'perturbation': r.cpu(),
        'perturbed_image': pert_image.cpu(),
        'original_label': orig_label,
        'adversarial_label': adv_label,
        'iterations': iterations,
        'true_label': target.item(),
        'l2_norm': torch.norm(r.cpu()).item(),
        'success': success
    })

    # Progress feedback
    print(f"  Example {idx+1}: True={target.item()}, Orig={orig_label}, "
          f"Adv={adv_label}, Iter={iterations}, L2={torch.norm(r.cpu()).item():.4f}")
```

The results dictionary captures nine distinct measurements per sample. Tensor data (`original_image`, `perturbation`, `perturbed_image`) gets transferred to CPU immediately via `.cpu()` to free GPU memory, preventing accumulation that would cause out-of-memory errors after processing dozens of samples. The L2 norm computation `torch.norm(r.cpu()).item()` measures total perturbation energy, with `.item()` converting from a single-element tensor to a Python float for storage. The progress print exposes patterns as they emerge: some samples flip with iterations of 1-2 and L2 norms around 2-3 (well-separated classes), while others require 5-6 iterations with L2 norms of 8-10 (ambiguous boundaries).

### Summarizing Attack Statistics

What does the population-level data show? We need aggregate statistics quantifying overall attack success, average perturbation magnitude, and typical convergence speed. These summary metrics enable comparing DeepFool’s behavior across different models, architectures, or datasets. The success rate indicates reliability, average L2 norm quantifies typical robustness, and average iteration count measures convergence efficiency.

Code: python

```python
print(f"\nAttack Success Rate: {success_count}/{num_examples} "
      f"({100*success_count/num_examples:.1f}%)")
print(f"Average L2 norm: {np.mean([r['l2_norm'] for r in results]):.4f}")
print(f"Average iterations: {np.mean([r['iterations'] for r in results]):.1f}")
```

Expected output:

Code: txt

```txt
Attack Success Rate: 20/20 (100.0%)
Average L2 norm: 4.8661
Average iterations: 3.0
```

The 100% success rate confirms DeepFool’s reliability on well-separated MNIST classes. The average L2 norm of 4.87 represents typical perturbation magnitude across diverse digit samples. The 3.0 average iterations indicate most samples converge in 2-4 steps, with variation between fast convergers (1 iteration for well-separated classes like ’3→5’) and slower ones (5 iterations for structurally similar classes like ’4→7’). Individual variation spans from L2 = 0.60 (digit ’3’ to ’5’, structurally similar) to L2 = 7.72 (digit ’7’ to ’2’, requiring significant structural changes). This 13x range demonstrates how decision boundary geometry varies widely across different regions of the input space, with some digit pairs separated by narrow gaps while others require substantial modifications.

## Visualizing Population-Level Attack Patterns

Statistics quantify attack effectiveness, but visual inspection demonstrates the human-perceptibility of minimal perturbations. Do adversarial examples look obviously corrupted, or do the changes remain subtle enough to avoid casual detection? We need a grid comparing multiple original-adversarial pairs side-by-side, enabling visual assessment of perturbation subtlety across different digit classes. The visualization should highlight successful attacks (classification changed) versus failures (classification unchanged), using our normal themeing. This provides qualitative validation of how minimal DeepFool's perturbations actually are.

Code: python

```python
def visualize_attack_grid(results, save_dir='output'):
    """
    Create grid visualization showing original and adversarial images side-by-side.

    Args:
        results (list): Attack results from batch generation
        save_dir (str): Directory to save visualization
    """
    print("\nGenerating attack grid visualization...")

    num_examples = min(10, len(results))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    for idx in range(num_examples):
        row = idx // 5
        col = idx % 5

        # Original image (top row for this column)
        ax_original = axes[row * 2, col]
        img = mnist_denormalize(results[idx]['original_image'].squeeze()).numpy()
        ax_original.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax_original.set_title(f"Original: {results[idx]['original_label']}",
                            color=HTB_GREEN, fontsize=10)
        ax_original.axis('off')

        # Adversarial image (bottom row for this column)
        ax_adv = axes[row * 2 + 1, col]
        adv_img = mnist_denormalize(results[idx]['perturbed_image'].squeeze()).numpy()
        ax_adv.imshow(adv_img, cmap='gray', vmin=0, vmax=1)

        title_color = MALWARE_RED if results[idx]['success'] else HACKER_GREY
        ax_adv.set_title(f"Adversarial: {results[idx]['adversarial_label']}",
                        color=title_color, fontsize=10)
        ax_adv.axis('off')

    plt.suptitle('DeepFool Attack: Original vs Adversarial Examples',
                color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_examples.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Grid visualization saved to {save_dir}/deepfool_examples.png")

# Generate the grid visualization
visualize_attack_grid(results, save_dir='output')
```

![Grid comparing MNIST digits before and after DeepFool, with original labels on top rows and adversarial labels like 7→2 and 0→6 in the bottom rows.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/deepfool_examples.png)

The grid layout uses integer division and modulo arithmetic for positioning: `row = idx // 5` determines which pair of rows (0 or 1, since each pair shows original/adversarial), while `col = idx % 5` determines the column position (0-4). Each original image occupies `axes[row * 2, col]`, and its adversarial counterpart sits directly below at `axes[row * 2 + 1, col]`. This pairing enables direct visual comparison per sample. Red titles (`MALWARE_RED`) mark successful classification flips, while grey titles indicate failures (rare on MNIST). The paired images demonstrate DeepFool's minimality: on MNIST's simple grayscale digits, careful inspection may reveal subtle differences, but they are only particularly obvious given the high contrast nature of the MNIST dataset (white on black). Examples like '7' flipped to '2', '9' to '4', and '5' to '6' maintain recognizable digit structure while completely fooling the classifier. On complex natural images (like photographs), such minimal perturbations would be nearly impossible to detect with the human eye.

