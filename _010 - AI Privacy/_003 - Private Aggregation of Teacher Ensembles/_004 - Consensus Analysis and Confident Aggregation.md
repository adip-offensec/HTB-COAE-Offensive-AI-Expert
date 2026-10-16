
# Consensus Analysis and Confident Aggregation

Not all pseudo-labels are created equal. When the ensemble reaches strong consensus, noise is unlikely to flip the prediction, producing reliable labels. When teachers are split, noise dominates, and the resulting label approaches random selection. Understanding this relationship enables a powerful optimization: selective labeling based on consensus.

Here we analyze teacher consensus patterns for the 10-class `MNIST` task with 250 teachers, introduce `confident aggregation` as a PATE-specific optimization, and demonstrate how filtering low-consensus samples improves label quality while reducing privacy expenditure.

## Understanding Teacher Consensus

Teacher consensus determines both label reliability and privacy cost. `High-consensus samples` (200+ votes) produce stable predictions that noise rarely flips. `Medium-consensus samples` (150-199 votes) occasionally flip but remain useful. `Low-consensus samples` (< 150 votes) represent genuinely ambiguous cases where noise dominates.

#### Consensus Categories

We categorize samples into three groups based on the maximum vote count. High-consensus samples (200+ votes, 80%+) experience noise almost never flipping predictions because margins are enormous. Medium-consensus samples (150-199 votes, 60-80%) occasionally experience flips. Low-consensus samples (fewer than 150 votes) represent genuinely ambiguous cases where even teachers disagree.

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 5: Teacher Consensus Analysis")
print("=" * 60)

num_teachers = len(teachers)
```

With the phase banner displayed, we compute consensus metrics by finding the maximum vote count for each sample:

Code: python

```python
# Calculate consensus metrics
max_votes = votes.max(axis=1)

high_threshold = 200  # 80% of 250 teachers
medium_threshold = 150  # 60% of 250 teachers

high_consensus = max_votes >= high_threshold
medium_consensus = (max_votes >= medium_threshold) & ~high_consensus
low_consensus = max_votes < medium_threshold

print(f"\nConsensus Distribution ({num_queries} samples):")
print(f"  High (≥80%):     {high_consensus.sum()} ({100*high_consensus.mean():.1f}%)")
print(f"  Medium (60-80%): {medium_consensus.sum()} ({100*medium_consensus.mean():.1f}%)")
print(f"  Low (<60%):      {low_consensus.sum()} ({100*low_consensus.mean():.1f}%)")
```

We compute `max_votes` as the highest vote count for any class on each sample. The boolean masks partition all samples into exactly one category.

#### Clean vs. Noisy Accuracy by Consensus

The relationship between consensus and label accuracy explains why selective labeling works:

Code: python

```python
# Calculate clean ensemble accuracy (no noise)
clean_labels = votes.argmax(axis=1)
clean_ensemble_acc = (clean_labels == y_query_true).mean()

# Calculate accuracy by consensus level
def accuracy_for_mask(mask, labels, true_labels):
    if mask.sum() == 0:
        return 0.0
    return (labels[mask] == true_labels[mask]).mean()

clean_high = accuracy_for_mask(high_consensus, clean_labels, y_query_true)
clean_medium = accuracy_for_mask(medium_consensus, clean_labels, y_query_true)
clean_low = accuracy_for_mask(low_consensus, clean_labels, y_query_true)

print(f"\nClean Ensemble Accuracy by Consensus:")
print(f"  High consensus:   {clean_high * 100:.2f}%")
print(f"  Medium consensus: {clean_medium * 100:.2f}%")
print(f"  Low consensus:    {clean_low * 100:.2f}%")
print(f"  Overall:          {clean_ensemble_acc * 100:.2f}%")
```

The pattern reflects confidence-correctness correlation. When 200+ teachers strongly agree on a digit, they are almost always correct, yielding ~98% accuracy for high-consensus samples. Medium consensus (150-199 votes) reflects genuine uncertainty, dropping to ~85% accuracy. Low-consensus samples approach random guessing at ~65% accuracy, revealing these are genuinely ambiguous digits (3 vs 8, 4 vs 9) where even expert ensembles struggle.

This has practical implications: low-consensus samples provide weak learning signal even before noise addition. After adding Laplacian noise, they become even less reliable, suggesting we should avoid labeling them entirely.

## The Impact of Noise

Laplacian noise affects samples differently based on their consensus strength. High-consensus samples (200+ votes) experience near-zero flip rates because large margins are virtually immune. Low-consensus samples flip more frequently but are rare in well-trained ensembles.

#### Noise Flip Analysis

To quantify how often noise changes predictions, we compare noisy labels against clean (no-noise) labels and measure the flip rate:

Code: python

```python
# Apply noisy aggregation
noise_scale = AGGREGATION_CONFIG['noise_scale']
noisy_labels = noisy_argmax(votes, noise_scale)

# Compare to clean predictions
noise_flipped = (noisy_labels != clean_labels)
flip_rate = noise_flipped.mean()

print(f"\nNoisy Aggregation Results (noise_scale={noise_scale}):")
print(f"  Labels matching true:  {(noisy_labels == y_query_true).mean() * 100:.2f}%")
print(f"  Labels matching clean: {(~noise_flipped).mean() * 100:.2f}%")
print(f"  Noise flip rate:       {flip_rate * 100:.2f}%")
```

This `noise_flipped` boolean array identifies samples where noise changed the prediction. Only a small fraction of predictions flip due to noise, reflecting the strong consensus achieved:

Code: python

```python
# Flip rate by consensus level
flip_high = noise_flipped[high_consensus].mean()
flip_medium = noise_flipped[medium_consensus].mean()
flip_low = noise_flipped[low_consensus].mean()

print(f"\nNoise Flip Rate by Consensus:")
print(f"  High consensus:   {flip_high * 100:.2f}%")
print(f"  Medium consensus: {flip_medium * 100:.2f}%")
print(f"  Low consensus:    {flip_low * 100:.2f}%")
```

High-consensus samples experience near-zero flip rates because large margins are virtually immune to noise. Low-consensus samples experience higher flip rates but are relatively rare in our dataset.

## Confident Aggregation

Confident aggregation exploits the consensus-accuracy relationship by labeling only samples where teachers strongly agree. This trades quantity for quality: fewer labeled samples, but more accurate labels and lower privacy cost. The technique rejects low-consensus samples before applying noisy argmax, assigning a sentinel value (-1) that we filter out before student training.

#### Implementation

We implement confident aggregation as a filter that rejects low-consensus samples before applying noisy argmax:

Code: python

```python
def confident_aggregation(votes, noise_scale, threshold):
    """Label only samples where teacher consensus exceeds threshold."""
    max_votes = votes.max(axis=1)
    confident_mask = max_votes >= threshold
    labels = np.full(len(votes), -1, dtype=np.int64)

    if confident_mask.sum() > 0:
        confident_votes = votes[confident_mask]
        noise = np.random.laplace(loc=0.0, scale=noise_scale, size=confident_votes.shape)
        labels[confident_mask] = np.argmax(confident_votes.astype(np.float64) + noise, axis=1)

    return labels, confident_mask
```

How does this filter work? We compute `max_votes` for each sample and reject any where the winning class received fewer than `threshold` votes. Rejected samples get label `-1` as a `sentinel value`, a special marker indicating "no label assigned" that we filter out before training the student. The value -1 works well because valid class labels are non-negative integers (0-9 for MNIST), making it trivially distinguishable. For accepted samples, we generate Laplacian noise and apply the same noisy argmax as before. With threshold 200, a sample where 180 teachers agree gets rejected, while one with 210 votes proceeds to noisy labeling.

#### Threshold Analysis

Different thresholds produce different tradeoffs between sample acceptance and label quality:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 6: Confident Aggregation Analysis")
print("=" * 60)

thresholds = [150, 175, 200, 215, 225]
```

We iterate through candidate thresholds, computing acceptance rate, label accuracy, and privacy savings for each:

Code: python

```python
print(f"\nConfident Aggregation Analysis:")
print(f"{'Threshold':<12} {'Accepted':<12} {'Accuracy':<12} {'ε Saved':<12}")
print("-" * 48)

for threshold in thresholds:
    labels, mask = confident_aggregation(votes, noise_scale, threshold)
    accepted_rate = mask.mean()

    if mask.sum() > 0:
        accuracy = (labels[mask] == y_query_true[mask]).mean() * 100
    else:
        accuracy = 0.0

    epsilon_saved = (1 - accepted_rate) * 100

    print(f"{threshold:<12} {100*accepted_rate:<12.1f}% {accuracy:<12.2f}% {epsilon_saved:<12.1f}%")
```

Lower thresholds accept more samples but with lower label quality. Higher thresholds accept fewer samples but produce more accurate labels and save privacy budget.

## Privacy Budget Savings

Each labeling query consumes privacy budget regardless of label quality. Confident aggregation saves budget by not labeling low-consensus samples, since rejected queries release no information and consume no privacy budget. With threshold 200 accepting ~85% of 5,000 queries, only 4,250 queries produce labels, and epsilon decreases proportionally.

#### Budget Calculation

We modify the privacy budget calculation to account for rejected queries, which consume no privacy budget because no label is released:

Code: python

```python
def compute_confident_privacy_budget(num_queries, noise_scale, acceptance_rate, delta=1e-5):
    """Compute privacy budget considering only accepted (labeled) queries."""
    per_query_eps = 2.0 / noise_scale
    effective_queries = int(num_queries * acceptance_rate)
    epsilon_sq_sum = effective_queries * (per_query_eps ** 2)
    total_epsilon = np.sqrt(2 * epsilon_sq_sum * np.log(1 / delta))
    total_epsilon += effective_queries * per_query_eps * (np.exp(per_query_eps) - 1)
    return total_epsilon, per_query_eps, effective_queries
```

Why do rejected queries consume no budget? No information about private data is released when we decline to label a sample. With threshold 200 accepting ~85% of 5,000 queries, only 4,250 queries actually produce labels. Privacy budget scales with effective queries, so ϵ decreases proportionally. This represents a privacy improvement while simultaneously improving label accuracy.

## Visualizing Consensus Effects

To make the consensus-accuracy relationship concrete, we visualize the distribution of maximum vote counts across all samples:

Code: python

```python
def plot_consensus_distribution(votes, save_path=None):
    """Plot histogram of maximum vote counts with threshold markers."""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    max_votes = votes.max(axis=1)
    ax.hist(max_votes, bins=range(100, 251, 10), color=AZURE, edgecolor=HACKER_GREY, alpha=0.8)
    ax.axvline(200, color=HTB_GREEN, linestyle='--', linewidth=2, label='High (≥200)')
    ax.axvline(150, color=AQUAMARINE, linestyle='--', linewidth=2, label='Medium (≥150)')
    ax.set_xlabel('Maximum Votes (out of 250 teachers)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Teacher Consensus Distribution', color=HTB_GREEN, fontweight='bold')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

We generate the histogram with vertical lines marking our consensus thresholds:

Code: python

```python
os.makedirs("figs", exist_ok=True)
plot_consensus_distribution(votes, save_path="figs/pate_consensus_distribution.png")
```

![Histogram showing distribution of maximum teacher votes across query samples. Most samples cluster in the 230-250 vote range (out of 250 teachers), indicating near-unanimous agreement. Dashed lines mark high consensus threshold at 200 votes (yellow) and medium at 150 votes (cyan).](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_consensus_distribution.png)

The histogram reveals why PATE achieves favorable privacy-utility tradeoffs on MNIST. Most samples cluster in the 230-250 vote range, indicating near-unanimous teacher agreement. The green dashed line marks the high-consensus threshold (200 votes), and cyan marks medium (150 votes). The vast majority of samples exceed even the strict threshold, explaining why noise scale 20 rarely flips predictions. When 230+ teachers agree, Laplacian noise would need to be exceptionally extreme to change the outcome.

## Analyzing Student Learning by Consensus

With consensus categories defined, we can analyze how the student model performs on samples grouped by original teacher agreement:

Code: python

```python
# Evaluate student on samples grouped by original consensus
student_preds = get_model_predictions(student_model, X_query, DEVICE).argmax(axis=1)

student_high = (student_preds[high_consensus] == y_query_true[high_consensus]).mean() * 100
student_medium = (student_preds[medium_consensus] == y_query_true[medium_consensus]).mean() * 100
student_low = (student_preds[low_consensus] == y_query_true[low_consensus]).mean() * 100

print(f"\nStudent Accuracy by Original Consensus:")
print(f"  High consensus samples:   {student_high:.2f}%")
print(f"  Medium consensus samples: {student_medium:.2f}%")
print(f"  Low consensus samples:    {student_low:.2f}%")
```

The pattern reveals how noise affects learning. On high-consensus samples, the student achieves excellent accuracy because it learned from mostly clean labels. On medium-consensus samples, accuracy drops slightly. On low-consensus samples, accuracy approaches random (10% for 10 classes) because training labels were noisier.

This stratification matches expectations: the student performs best on samples where teachers agreed, which aligns with the training signal it received. The student cannot learn to classify digits that the teachers themselves could not agree on.

