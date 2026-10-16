
# Noisy Aggregation and Privacy Guarantees

Adding calibrated noise during the aggregation of teacher votes is what gives PATE its privacy guarantees. Without this noise, an attacker could potentially infer membership information by analyzing patterns in the ensemble's predictions. Here we implement the noisy aggregation mechanism, explain how it provides differential privacy, and generate the pseudo-labeled data for student training.

## The Laplacian Mechanism

We add Laplacian noise to each class vote count before selecting the winner. The `Laplacian distribution` is centered at zero with scale parameter b, and its probability density function is:

f(x∣b)=12bexp⁡(−|x|b)

With a noise scale of 20.0, a draw from Lap(20) has standard deviation 202≈28.3. This noise magnitude is moderate relative to the strong consensus achievable with our teacher ensemble. The aggregation selects the class with the highest noisy vote count:

ŷ=arg⁡max⁡j(nj(x)+Lap(scale))

Here nj(x) is the number of teachers voting for class j on input x.

#### Implementing Noisy Aggregation

To add noise and select the winning class, we implement a function that takes vote counts and returns noisy pseudo-labels:

Code: python

```python
def noisy_argmax(votes, noise_scale):
    """Add Laplacian noise to votes and return the winning class."""
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=votes.shape)
    noisy_votes = votes.astype(np.float64) + noise
    return np.argmax(noisy_votes, axis=1)
```

How does this work mechanically? For 5,000 samples with 10 classes, `np.random.laplace` generates a `(5000, 10)` noise matrix where each entry is an independent draw from the Laplacian distribution. We convert integer vote counts to float64 before addition to preserve decimal precision. Consider a sample where 200 teachers vote class 3 and 35 vote class 7: after adding noise draws of, say, -8.2 and +12.1, the noisy counts become 191.8 and 47.1, preserving the consensus. Only when margins are small can noise flip the outcome.

#### Privacy Through Noise

Adding noise obscures the contribution of any single training sample. Consider two neighboring datasets that differ in exactly one sample. In PATE, this means one teacher was trained on slightly different data. The vote counts for that teacher may differ, but all other 249 teachers remain unchanged. Changing one vote in a 250-teacher ensemble causes a small change in the vote histogram: at most +1 for one class and -1 for another.

The Laplacian mechanism ensures that an observer cannot reliably distinguish between the original vote counts and any counts that differ by a small amount. Formally, for neighboring datasets D and D′, the ratio of probabilities of any output ŷ is bounded. This ratio captures how much the output distribution can shift when a single training sample changes: if the ratio is close to 1, the mechanism reveals little about any individual sample.

P[ℳ(D)=ŷ]P[ℳ(D′)=ŷ]≤eϵ

This is the definition of ϵ-differential privacy. The sensitivity of our vote aggregation (maximum change from modifying one sample) is 2 because one vote shifts from one class to another. The privacy guarantee for a single query is therefore ϵ=2noise_scale. With noise scale 20, each query provides ϵ=0.10 privacy.

## Privacy Budget Composition

Each labeling query consumes some privacy budget. To train the student model, we need many labeled samples, so understanding how privacy costs compose is essential.

#### Basic and Advanced Composition

The simplest composition theorem tells us that k queries, each with privacy cost ϵ0, together provide (k⋅ϵ0)-differential privacy. With 5,000 queries at ϵ0=0.10, naive composition gives ϵ=500, which represents negligible privacy protection. Fortunately, this linear scaling is pessimistic. The advanced composition theorem provides a tighter bound that grows with k instead of k:

Code: python

```python
def compute_privacy_budget(num_queries, noise_scale, delta=1e-5):
    """
    Compute total privacy budget using advanced composition.

    Args:
        num_queries (int): Number of labeling queries made.
        noise_scale (float): Laplacian noise scale.
        delta (float): Failure probability.

    Returns:
        tuple: (total_epsilon, per_query_epsilon)
    """
    per_query_eps = 2.0 / noise_scale
    epsilon_sq_sum = num_queries * (per_query_eps ** 2)
    total_epsilon = np.sqrt(2 * epsilon_sq_sum * np.log(1 / delta))
    total_epsilon += num_queries * per_query_eps * (np.exp(per_query_eps) - 1)
    return total_epsilon, per_query_eps
```

We compute ϵ using the advanced composition formula. The first term 2⋅k⋅ϵ02⋅ln⁡(1/δ) captures the dominant k scaling. The second term adds a correction for finite ϵ0. With our configuration of 5,000 queries and noise scale 20.0, we get per-query ϵ0=0.10 and total ϵ≈8.81, representing strong privacy protection.

## Generating Pseudo-Labeled Data

With the noisy aggregation mechanism defined, we generate training data for the student model. This phase spends the privacy budget. First, we select which public samples to label:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 3: Noisy Aggregation and Privacy Budget")
print("=" * 60)

num_queries = min(AGGREGATION_CONFIG['num_student_queries'], len(X_public_norm))
query_indices = np.random.choice(len(X_public_norm), num_queries, replace=False)
X_query = X_public_norm[query_indices]
y_query_true = y_public[query_indices]
```

We randomly select 5,000 samples from the 12,000 available public samples without replacement. The `y_query_true` variable holds the ground truth labels for later evaluation (the student never sees these). Now we collect votes and apply noisy aggregation:

Code: python

```python
votes = get_teacher_votes(teachers, X_query, DEVICE)
student_labels = noisy_argmax(votes, AGGREGATION_CONFIG['noise_scale'])

label_accuracy = (student_labels == y_query_true).mean() * 100
clean_preds = votes.argmax(axis=1)
clean_ensemble_acc = (clean_preds == y_query_true).mean() * 100

print(f"Labeled {num_queries} public samples")
print(f"Clean ensemble accuracy: {clean_ensemble_acc:.2f}%")
print(f"Noisy label accuracy: {label_accuracy:.2f}%")
```

Here `label_accuracy` measures how many noisy pseudo-labels match ground truth, indicating the training signal quality the student receives. Meanwhile, `clean_ensemble_acc` shows accuracy without noise, revealing privacy's cost. Typical results show clean ensemble at ~88% and noisy labels at ~87%, a gap of less than 1 percentage point.

#### Analyzing Label Quality

How much information does the noise mechanism obscure? We can measure this by comparing clean ensemble accuracy to noisy label accuracy. With 250 teachers on the 10-class `MNIST` task, the gap is remarkably small because teachers achieve exceptionally strong consensus:

Code: python

```python
# Compute privacy budget
total_eps, per_query_eps = compute_privacy_budget(
    num_queries, AGGREGATION_CONFIG['noise_scale']
)

print(f"\nPrivacy budget:")
print(f"  Per-query epsilon: {per_query_eps:.4f}")
print(f"  Total epsilon: {total_eps:.2f}")
```

## Effect of Noise on Label Quality

Different noise levels produce different privacy-utility tradeoffs. Higher noise provides stronger privacy but reduces label accuracy. At our default noise scale of 20.0, we achieve approximately 87% label accuracy while maintaining strong privacy protection. The effect of noise depends heavily on teacher consensus: when teachers strongly agree on a prediction, even substantial noise rarely flips the outcome. A later section analyzes consensus patterns in detail, showing how to exploit strong agreement for improved label quality.

## Visualizing Aggregation Effects

To make the aggregation mechanism's behavior concrete, we create a bar chart comparing accuracy at each stage:

Code: python

```python
def plot_accuracy_comparison(models, accuracies, save_path=None):
    """Plot comparison of model accuracies."""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    colors = [MALWARE_RED, AZURE, HTB_GREEN, AQUAMARINE]
    bars = ax.bar(models, accuracies, color=colors, edgecolor=HACKER_GREY, linewidth=2)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', color=WHITE, fontweight='bold')

    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', color=HTB_GREEN, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

Now we compute the average individual teacher accuracy and generate the comparison:

Code: python

```python
avg_teacher_acc = np.mean([evaluate_accuracy(t, holdout_loader, DEVICE) for t in teachers]) / 100
os.makedirs("figs", exist_ok=True)
plot_accuracy_comparison(
    models=['Avg Teacher', 'Clean Ensemble', 'Noisy Labels'],
    accuracies=[avg_teacher_acc, clean_ensemble_acc / 100, label_accuracy / 100],
    save_path="figs/pate_accuracy_comparison.png"
)
```

![Bar chart comparing model accuracy across three configurations. Average individual teacher achieves 80.2% (red), clean ensemble voting achieves 87.8% (blue), and noisy label aggregation achieves 87.2% (green). Shows ensemble power and minimal privacy cost.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_accuracy_comparison.png)

What does this progression reveal? Ensemble aggregation's power and privacy's modest cost. Individual teachers average ~80% accuracy, constrained by their small 192-sample partitions. When teachers vote together, the clean ensemble reaches ~88%, a dramatic improvement demonstrating how aggregation overcomes individual weakness. Adding Laplacian noise reduces accuracy by less than 1 percentage point to ~87%. For `MNIST` with 250 teachers achieving strong consensus, the privacy cost is remarkably low.

