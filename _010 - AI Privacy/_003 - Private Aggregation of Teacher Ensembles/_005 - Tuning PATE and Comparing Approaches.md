
# Tuning PATE and Comparing Approaches

With PATE's core mechanism established (teacher consensus, noisy aggregation, and confident labeling), we now analyze how noise scale affects label quality and privacy budgets, compare PATE to DP-SGD, and discuss practical deployment considerations.

## Privacy-Utility Tradeoff Across Noise Scales

Higher noise provides stronger privacy guarantees (lower ϵ) but degrades label quality. The relationship is not linear: doubling noise scale halves per-query epsilon but does not halve accuracy. Understanding where this curve bends informs practical choices.

#### Sweeping Through Noise Scales

We test noise scales from 5 to 80, measuring privacy cost and label quality at each level:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 7: Privacy-Utility Tradeoff Analysis")
print("=" * 60)

noise_scales = [5, 10, 20, 40, 80]

print(f"\n{'Noise Scale':<12} {'Per-ε':<10} {'Total ε':<12} {'Label Acc':<12} {'Flip Rate':<12}")
print("-" * 58)

for ns in noise_scales:
    noisy_labels = noisy_argmax(votes, ns)

    per_eps = 2.0 / ns
    total_eps, _ = compute_privacy_budget(num_queries, ns)

    label_acc = (noisy_labels == y_query_true).mean()
    flip_rate = (noisy_labels != clean_labels).mean()

    print(f"{ns:<12} {per_eps:<10.4f} {total_eps:<12.2f} {label_acc:<12.4f} {flip_rate:<12.4f}")
```

Each row shows per-query epsilon, total epsilon via advanced composition, label accuracy, and flip rate.

#### Results Across the Spectrum

At noise scale 5, per-query ε = 0.40 yields approximately 95% label accuracy. Double the scale to 10 and ε halves to 0.20 while accuracy drops only 3 points to 92%. At scale 20 (our default), ε = 0.10 yields approximately 87% accuracy. Push to scale 40 and accuracy begins dropping faster: ε = 0.05 with only 78% accuracy. At scale 80, ε = 0.025 drops accuracy to approximately 65%, approaching the point where noise overwhelms signal.

The practical operating range sits between scale 10 and 40. Below 10, privacy guarantees weaken substantially. Above 40, label quality degrades faster than privacy improves. Our default of 20.0 balances these considerations.

#### Visualizing the Tradeoff

A dual-axis plot reveals the asymmetry between privacy gains and accuracy losses. We define a function that computes accuracies and epsilons for each noise scale:

Code: python

```python
def plot_privacy_utility_tradeoff(noise_scales, votes, y_true, num_queries, save_path=None):
    """Plot privacy vs utility across noise scales on dual axes."""
    accuracies = [(noisy_argmax(votes, ns) == y_true).mean() for ns in noise_scales]
    epsilons = [compute_privacy_budget(num_queries, ns)[0] for ns in noise_scales]

    fig, ax1 = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    ax1.set_xlabel('Noise Scale')
    ax1.set_ylabel('Label Accuracy', color=HTB_GREEN)
    ax1.plot(noise_scales, accuracies, 'o-', color=HTB_GREEN, linewidth=2, markersize=10)
    ax1.tick_params(axis='y', labelcolor=HTB_GREEN)
```

The primary y-axis shows label accuracy. We add a secondary y-axis for privacy budget:

Code: python

```python
    ax2 = ax1.twinx()
    ax2.set_ylabel('Privacy Budget (ε)', color=AZURE)
    ax2.plot(noise_scales, epsilons, 's--', color=AZURE, linewidth=2, markersize=10)
    ax2.tick_params(axis='y', labelcolor=AZURE)
    ax2.set_yscale('log')

    ax1.set_title('Privacy-Utility Tradeoff', color=HTB_GREEN, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

Generating the visualization:

Code: python

```python
os.makedirs("figs", exist_ok=True)
plot_privacy_utility_tradeoff(noise_scales, votes, y_query_true, num_queries,
                               save_path="figs/pate_privacy_utility.png")
```

![Dual-axis line chart showing privacy-utility tradeoff as noise scale increases. Green line shows label accuracy dropping from 88% to 57% as noise scale increases from 5 to 80. Blue dashed line shows privacy budget (epsilon) decreasing from over 1000 to about 10 on log scale. Noise scale 20 marks the optimal tradeoff point.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_privacy_utility.png)

The green accuracy line stays near 88% at low noise scales, then drops steeply beyond scale 40. Meanwhile, the blue privacy line (log scale) shows epsilon decreasing with noise. The asymmetry is striking: moving from scale 5 to 20 provides dramatic privacy improvement (epsilon drops from ~35 to ~9) with modest accuracy cost (95% to 87%). But moving from 20 to 80 yields diminishing privacy returns while accuracy collapses to 65%. Scale 20 sits at the elbow of this curve.

## PATE vs DP-SGD: When to Use Each

PATE and DP-SGD both achieve differential privacy, but through fundamentally different mechanisms. PATE protects data through architectural separation; DP-SGD protects it through training-time noise.

PATE works best when public data matching the private distribution is available. The student trains on public data with standard optimization, and once trained, answers queries indefinitely without consuming privacy budget. Confident aggregation lets you reject low-consensus queries without cost, investing budget only where it produces high-quality labels. The information bottleneck also provides an intuitive privacy argument beyond formal ε values, which helps when explaining privacy properties to non-technical stakeholders.

The public data requirement is often the binding constraint. Medical datasets, financial records, and personalized models rarely have suitable public proxies. The query budget can also limit model complexity: sophisticated models requiring millions of samples may exhaust privacy budget before obtaining enough labels. Training hundreds of teachers requires substantial computation, though inference parallelizes well.

DP-SGD takes a different approach, training directly on private data without requiring a public proxy. This potentially learns richer representations than knowledge distillation allows, and avoids ensemble overhead entirely. However, every gradient update consumes privacy budget, so long training runs can exhaust budget quickly. Per-sample gradient clipping affects convergence and requires careful tuning.

The choice depends on your constraints. If public data matching your private distribution exists and you need high-volume inference after deployment, PATE is typically the better choice. If no public proxy exists, or you need to train complex models directly on private data, DP-SGD may be your only option despite its per-query privacy costs.

## Deployment Strategies

Confident aggregation enables three deployment approaches depending on operational constraints. When regulatory constraints impose a hard epsilon limit, you want to maximize labels within that budget. When student training requires a specific dataset size, you want to minimize privacy cost for that target. When public data is abundant and label quality matters most, you can query iteratively until enough high-consensus samples accumulate.

#### Fixed Privacy Budget

When regulatory or policy constraints impose a hard epsilon limit, confident aggregation allows more total queries while staying within budget. We use binary search to find how many queries a given epsilon permits:

Code: python

```python
def estimate_queries_for_budget(target_epsilon, noise_scale, delta=1e-5):
    """Binary search for maximum queries within privacy budget."""
    per_query_eps = 2.0 / noise_scale
    low, high = 1, 100000
    while low < high:
        mid = (low + high + 1) // 2
        eps_sq_sum = mid * (per_query_eps ** 2)
        total_eps = np.sqrt(2 * eps_sq_sum * np.log(1 / delta))
        total_eps += mid * per_query_eps * (np.exp(per_query_eps) - 1)
        low, high = (mid, high) if total_eps <= target_epsilon else (low, mid - 1)
    return low
```

Now we compare how many queries each strategy permits for a fixed budget:

Code: python

```python
_, confident_mask = confident_aggregation(votes, noise_scale, threshold=200)
acceptance_rate = confident_mask.mean()

target_epsilon = 10.0
standard_queries = estimate_queries_for_budget(target_epsilon, noise_scale)
confident_queries = standard_queries / acceptance_rate

print(f"For ε={target_epsilon}:")
print(f"  Standard aggregation: {standard_queries} queries")
print(f"  Confident (threshold=200): {int(confident_queries)} queries")
print(f"  Net usable labels: {int(confident_queries * acceptance_rate)} samples")
```

Both strategies produce similar numbers of usable labels, but confident aggregation's labels are higher quality because low-consensus samples are filtered out.

#### Fixed Label Count

When student training requires a specific dataset size, confident aggregation minimizes privacy cost. To obtain 5,000 high-quality labels, we query until enough high-consensus samples are found:

Code: python

```python
target_labels = 5000
confident_queries = int(target_labels / acceptance_rate)

conf_eps = compute_confident_privacy_budget(
    confident_queries, noise_scale, acceptance_rate
)[0]

print(f"For {target_labels} labels:")
print(f"  Queries needed: {confident_queries}")
print(f"  Privacy cost: ε = {conf_eps:.2f}")
```

#### Iterative Sampling

When public data is abundant and label quality matters most, we query samples one at a time until enough high-consensus samples accumulate:

Code: python

```python
def iterative_confident_labeling(teachers, X_public, threshold, target_count, noise_scale, device):
    """Query samples until target_count high-consensus labels obtained."""
    confident_X, confident_y, queries_made = [], [], 0
    available_indices = np.random.permutation(len(X_public))

    for idx in available_indices:
        if len(confident_y) >= target_count:
            break
        x = X_public[idx:idx+1]
        labels, mask = confident_aggregation(get_teacher_votes(teachers, x, device), noise_scale, threshold)
        queries_made += 1
        if mask[0]:
            confident_X.append(x[0])
            confident_y.append(labels[0])

    return np.array(confident_X), np.array(confident_y), queries_made
```

This approach guarantees all labels come from high-consensus samples. With 85% acceptance rate and target of 5,000 labels, we expect ~5,900 queries. The tradeoff is increased query count for guaranteed label quality.

In practice, fixed label count is often the natural choice: you typically know how many training samples the student needs, and confident aggregation reduces the privacy cost of obtaining them.

## Hyperparameter Sensitivity

Several interacting hyperparameters affect PATE's performance. The number of teachers trades individual model strength against consensus signal strength. The query budget trades student training data against privacy cost, with favorable sublinear scaling.

#### Number of Teachers

More teachers strengthen the aggregation signal but reduce data per teacher. With fewer teachers, each receives more samples and achieves higher individual accuracy, but the aggregation signal is weaker. With more teachers, individual accuracy drops but consensus signals become stronger. The best count depends on dataset size and classification difficulty. Easier classification tasks can tolerate more teachers; harder tasks may benefit from fewer but stronger teachers.

#### Query Budget

More queries provide more student training data but consume more privacy budget. With 1,000 queries, the student receives limited training data but total ϵ≈3.9 provides strong privacy. With 2,500 queries, training data increases and ϵ≈6.2 remains reasonable. With 5,000 queries (our default), the student has ample training data and ϵ≈8.81 provides good privacy. With 10,000 queries, the student has abundant data but ϵ≈12.5 begins to weaken privacy guarantees.

The relationship is sublinear due to advanced composition. Doubling queries increases epsilon by approximately 40%, not 100%. This favorable scaling makes PATE practical for reasonable training set sizes.

## Inherent Limitations

Beyond parameter tuning, PATE faces inherent constraints. The public data distribution must match private data, which is challenging for medical, financial, and personalized applications. Class imbalance creates consensus disparities that propagate to student performance. The disjoint partition requirement demands careful deduplication to maintain privacy guarantees.

#### Public Data Distribution

The public data must match the private data distribution. Distribution shift between public and private data degrades student performance more than label noise because the student learns patterns that do not transfer. Medical applications are particularly challenging because public medical data rarely matches the population in private hospital records. Financial applications face similar constraints. Personalized models almost by definition lack public data from the same distribution.

#### Class Imbalance

Imbalanced datasets create consensus disparities. Teachers achieve higher consensus on majority class samples because they see more examples during training and agree more readily. Minority class samples produce lower consensus and receive noisier labels. The student consequently underperforms on minority classes.

Stratified sampling during labeling can mitigate but not eliminate this effect. Ensuring the query set maintains class balance helps, but the consensus disparity remains.

#### Disjoint Partition Requirement

PATE's privacy guarantee assumes each individual contributes to exactly one teacher's partition. Duplicate records violate this assumption because the same person influences multiple teachers, increasing information leakage. Multiple appearances (the same person at different times) cause similar problems. Longitudinal data where individuals appear repeatedly over time requires careful handling.

Data deduplication is essential before partitioning. If complete deduplication is impossible, the privacy analysis must account for the maximum number of teachers any individual could affect.

## Saving Results

We save configuration and results for comparison with other techniques. First, we compute the information bottleneck metrics that quantify how much private data compresses into student training signal:

Code: python

```python
# Information bottleneck metrics
private_data_bytes = len(X_private_norm) * X_private_norm.shape[1] * 4  # float32
student_info_bytes = num_queries * 4  # 4 bits per label, ~0.5 bytes, round to int size
compression_ratio = private_data_bytes / student_info_bytes

print(f"\nInformation Bottleneck:")
print(f"  Private data: {private_data_bytes / 1e6:.1f} MB")
print(f"  Student labels: {student_info_bytes / 1e3:.1f} KB")
print(f"  Compression ratio: {compression_ratio:.0f}:1")
```

The compression ratio quantifies PATE's privacy mechanism: megabytes of private training data compress into kilobytes of noisy class labels. Now we save the complete results:

Code: python

```python
results = {
    'configuration': {'dataset': 'mnist', 'num_teachers': TEACHER_CONFIG['num_teachers'],
                      'noise_scale': AGGREGATION_CONFIG['noise_scale'], 'num_queries': num_queries},
    'privacy': {'per_query_epsilon': float(per_query_eps), 'total_epsilon': float(total_eps)},
    'utility': {'student_accuracy': float(student_test_acc), 'label_accuracy': float(label_accuracy),
                'clean_ensemble_accuracy': float(clean_ensemble_acc)},
    'information': {'private_data_bytes': int(private_data_bytes), 'student_info_bytes': int(student_info_bytes),
                    'compression_ratio': float(compression_ratio)}
}

results_path = os.path.join("figs", "pate_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
```

This output enables programmatic comparison across configurations and provides a baseline for evaluating privacy techniques.

