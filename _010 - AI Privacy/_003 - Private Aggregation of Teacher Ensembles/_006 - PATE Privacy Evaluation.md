
# PATE Privacy Evaluation

Privacy guarantees can be stated mathematically through differential privacy, but empirical validation matters for practical deployments. Membership inference attacks provide a concrete test: can an attacker determine whether specific individuals were in the training data by observing model behavior? Here we evaluate PATE's resistance to such attacks, demonstrating that the architectural separation between teachers and students provides meaningful privacy protection beyond theoretical bounds.

## Training the Vulnerable Baseline

Before evaluating PATE's privacy, we train a vulnerable baseline model to establish what unprotected models leak. This model trains directly on sensitive data without any privacy protection, serving as the reference point for attack comparison:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 8: Privacy Evaluation")
print("=" * 60)

BASELINE_CONFIG = {
    "hidden_layers": [256, 128],
    "dropout": 0.0,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "train_size": 10000,
}
```

We follow standard practice for demonstrating `MIA` vulnerability. The `[256, 128]` architecture is appropriately sized for the 10-class digit classification task. Zero dropout maximizes overfitting, making the model more vulnerable to attacks. We train on 10,000 samples to produce reliable attack accuracy measurements.

Code: python

```python
# Select subset of private data for baseline training
np.random.seed(RANDOM_SEED)
baseline_indices = np.random.choice(
    len(X_private_norm), BASELINE_CONFIG['train_size'], replace=False
)
X_baseline_members = X_private_norm[baseline_indices]
y_baseline_members = y_private[baseline_indices]

# Track which samples are NOT in the baseline training set
baseline_nonmember_mask = np.ones(len(X_private_norm), dtype=bool)
baseline_nonmember_mask[baseline_indices] = False
X_baseline_nonmembers = X_private_norm[baseline_nonmember_mask]
y_baseline_nonmembers = y_private[baseline_nonmember_mask]

print(f"Baseline members: {len(X_baseline_members)} samples")
print(f"Baseline non-members: {len(X_baseline_nonmembers)} samples")
```

We now have the member/non-member split needed for fair `MIA` evaluation. Both sets come from the same distribution (private `MNIST` data), differing only in whether they were used for training. With 10,000 members selected from 48,000 private samples, the remaining 38,000 serve as non-members.

#### Training the Baseline Model

With data prepared, we train the baseline model using the holdout set for validation:

Code: python

```python
baseline_member_loader = create_dataloader(
    X_baseline_members, y_baseline_members,
    BASELINE_CONFIG['batch_size']
)

baseline_model = MLP(
    input_size=num_features,
    hidden_layers=BASELINE_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=BASELINE_CONFIG['dropout']
)

train_model(
    baseline_model, baseline_member_loader, holdout_loader,
    device=DEVICE,
    epochs=BASELINE_CONFIG['epochs'],
    learning_rate=BASELINE_CONFIG['learning_rate']
)
```

We train for 50 epochs with zero dropout, deliberately encouraging overfitting. After training, the model achieves high accuracy on its training data but shows measurable overfitting that membership inference attacks can exploit.

## Implementing the Membership Inference Attack

Our attack exploits a consistent observation: models behave differently on samples they have memorized versus samples they have not seen. Specifically, models tend to produce higher-confidence predictions on training data because they have optimized specifically for those examples. We implement a `confidence threshold attack` that classifies samples as members if their prediction confidence exceeds a threshold:

Code: python

```python
def compute_mia_advantage(model, X_members, y_members, X_nonmembers, y_nonmembers, device):
    """
    Compute membership inference attack advantage using confidence threshold.
    """
    member_probs = get_model_predictions(model, X_members, device)
    member_confidence = np.max(member_probs, axis=1)

    nonmember_probs = get_model_predictions(model, X_nonmembers, device)
    nonmember_confidence = np.max(nonmember_probs, axis=1)
```

We extract the maximum softmax probability for each sample as a confidence score. A sample with prediction `[0.95, 0.05, ...]` has confidence 0.95, while one with `[0.55, 0.25, ...]` has confidence 0.55. Our hypothesis is that members will cluster at higher confidence values because the model has seen and memorized them.

#### Finding the Best Threshold

To separate member and non-member confidence distributions, we search over percentiles to find the threshold that maximizes classification accuracy:

Code: python

```python
    n_samples = min(len(member_confidence), len(nonmember_confidence))
    member_conf_balanced = member_confidence[:n_samples]
    nonmember_conf_balanced = nonmember_confidence[:n_samples]

    all_confidence = np.concatenate([member_conf_balanced, nonmember_conf_balanced])
    all_labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    thresholds = np.percentile(all_confidence, np.linspace(0, 100, 1000))
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (all_confidence >= threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    attack_advantage = best_accuracy - 0.5
    return best_accuracy, attack_advantage, best_threshold
```

Why balance the datasets? With equal members and non-members, a random classifier achieves exactly 50% accuracy. This makes 50% the natural baseline for measuring attack success. If we used unbalanced sets (say 90% members), even a trivial "always predict member" strategy would achieve 90% accuracy, obscuring the attack's true effectiveness.

Why search over percentiles? Percentiles span the full range of observed confidence values. Using `np.percentile(all_confidence, np.linspace(0, 100, 1000))` generates 1,000 candidate thresholds from the minimum to maximum observed confidence. This exhaustive search finds the threshold that best separates the two distributions, giving the attacker every advantage. If the model leaks membership information, this search will find the optimal way to exploit it.

We try each candidate threshold and count correct classifications. For threshold 0.85, any sample with confidence ≥ 0.85 is predicted as "member" and any below as "non-member." The threshold yielding highest accuracy becomes the attack's decision boundary. The attack advantage is accuracy minus 0.5: an accuracy of 55% yields advantage 0.05 (5 percentage points above random guessing). This metric directly measures how much information the model leaks about its training set.

## Attacking the Baseline Model

With the attack implemented, we evaluate the vulnerable baseline:

Code: python

```python
print("\n--- Attacking Baseline Model ---")
print(f"  Members: {len(X_baseline_members)} samples (baseline training data)")
print(f"  Non-members: {len(X_baseline_nonmembers)} samples")

baseline_mia_acc, baseline_mia_adv, baseline_threshold = compute_mia_advantage(
    baseline_model,
    X_baseline_members, y_baseline_members,
    X_baseline_nonmembers, y_baseline_nonmembers,
    DEVICE
)

print(f"  Attack accuracy:  {baseline_mia_acc:.4f}")
print(f"  Attack advantage: {baseline_mia_adv:.4f}")
```

Our attack achieves accuracy above random guessing, demonstrating that the unprotected model leaks membership information. An attacker observing only predictions can determine with some accuracy whether specific digit images were used for training. While `MNIST` images are less sensitive than medical records, this demonstrates the attack methodology that would apply to more sensitive data.

## Attacking the PATE Student

The question is whether the PATE student, which never directly accessed private data, leaks information about that data through its predictions. We test this by attacking the student with the private data as the member set:

Code: python

```python
print("\n--- Attacking PATE Student Model ---")

student_mia_acc_private, student_mia_adv_private, _ = compute_mia_advantage(
    student_model,
    X_private_norm, y_private,
    X_holdout_norm, y_holdout,
    DEVICE
)

print(f"  Attack accuracy:  {student_mia_acc_private:.4f}")
print(f"  Attack advantage: {student_mia_adv_private:.4f}")
```

We achieve approximately 50% accuracy with near-zero advantage, essentially indistinguishable from random guessing. The student model cannot distinguish private data from holdout data because it never trained on the private data directly. Its knowledge comes exclusively from the noisy pseudo-labels generated by the teacher ensemble.

#### Understanding the Protection Mechanism

Why does PATE resist membership inference? The protection arises from architectural design, not just noise addition. The student model has no direct connection to private data samples. Information flows through a severe bottleneck: 48,000 private digit images with 784 features each compress into just 5,000 noisy class labels. Each label is a single integer (0-9) determined by noisy vote aggregation. Attempting to infer individual membership from these labels is like trying to identify a specific drop of water after it has mixed with a river.

We can also test whether the student leaks information about the public query samples it trained on directly:

Code: python

```python
student_mia_acc_query, student_mia_adv_query, _ = compute_mia_advantage(
    student_model,
    X_query, student_labels,
    X_holdout_norm, y_holdout,
    DEVICE
)

print(f"  Attack on query data accuracy:  {student_mia_acc_query:.4f}")
print(f"  Attack on query data advantage: {student_mia_adv_query:.4f}")
```

Here we achieve higher accuracy with measurable advantage. This leakage about the query data is expected and not a privacy concern: the student did train directly on these samples, and query data is public by assumption. The high attack accuracy on query data actually confirms our methodology works correctly. This leakage concerns only the public query data, not the private data that PATE aims to protect.

## Comparing Attack Results

Comparing results reveals PATE's protective effect:

Code: python

```python
print("\n" + "-" * 50)
print("SUMMARY: Attack Advantage Comparison")
print("-" * 50)
print(f"  Baseline (no protection):    {baseline_mia_adv:.4f}")
print(f"  PATE Student (private data): {student_mia_adv_private:.4f}")
print(f"  PATE Student (query data):   {student_mia_adv_query:.4f}")

if baseline_mia_adv > 0:
    reduction = (baseline_mia_adv - student_mia_adv_private) / baseline_mia_adv * 100
    print(f"\n  Protection improvement: {reduction:.1f}%")
```

The baseline leaks membership information (attack advantage above random), while the PATE student leaks essentially none about the private data (advantage near zero, statistically indistinguishable from random guessing). We see substantial reduction in attack advantage. The remaining small advantage likely reflects random variation, not true leakage.

## Visualizing Confidence Distributions

Why does the attack succeed on the baseline? Member and non-member confidence distributions differ. We visualize this separation by plotting the density difference between member and non-member confidence values. First, we define a function to plot confidence distribution differences:

Code: python

```python
def plot_confidence_distributions(baseline_member, baseline_nonmember,
                                   student_private, student_holdout, save_path=None):
    """Plot density difference between member and non-member confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline model
    bins = np.linspace(0.5, 1.0, 50)
    member_hist, _ = np.histogram(baseline_member, bins=bins, density=True)
    nonmember_hist, _ = np.histogram(baseline_nonmember, bins=bins, density=True)
    diff = member_hist - nonmember_hist
    bin_centers = (bins[:-1] + bins[1:]) / 2

    axes[0].bar(bin_centers, diff, width=0.01, color=MALWARE_RED, alpha=0.8)
    axes[0].axhline(y=0, color=WHITE, linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Density Difference (Member - Non-member)')
    axes[0].set_title('Baseline Model', color=MALWARE_RED, fontweight='bold')
```

This creates histograms for both member and non-member confidence values, then plots their difference. We apply the same approach to the PATE student:

Code: python

```python
    # PATE student
    student_member_hist, _ = np.histogram(student_private, bins=bins, density=True)
    student_nonmember_hist, _ = np.histogram(student_holdout, bins=bins, density=True)
    student_diff = student_member_hist - student_nonmember_hist

    axes[1].bar(bin_centers, student_diff, width=0.01, color=HTB_GREEN, alpha=0.8)
    axes[1].axhline(y=0, color=WHITE, linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Density Difference (Private - Holdout)')
    axes[1].set_title('PATE Student', color=HTB_GREEN, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

We also define a function to create a bar chart comparing attack results across models:

Code: python

```python
def plot_mia_comparison(baseline_adv, student_adv_private, student_adv_query,
                        baseline_acc, student_acc_private, student_acc_query, save_path=None):
    """Plot bar chart comparing MIA results across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Baseline', 'PATE\n(Private)', 'PATE\n(Query)']
    advantages = [baseline_adv, student_adv_private, student_adv_query]
    accuracies = [baseline_acc, student_acc_private, student_acc_query]
    colors = [MALWARE_RED, HTB_GREEN, AQUAMARINE]

    axes[0].bar(labels, advantages, color=colors, edgecolor=HACKER_GREY, linewidth=2)
    axes[0].axhline(y=0.05, color=NUGGET_YELLOW, linestyle='--', linewidth=2, label='5% threshold')
    axes[0].set_ylabel('Attack Advantage')
    axes[0].set_title('MIA Attack Advantage', color=HTB_GREEN, fontweight='bold')
    axes[0].legend()
```

The first subplot shows attack advantage with a 5% threshold line. We add the second subplot for attack accuracy:

Code: python

```python
    axes[1].bar(labels, accuracies, color=colors, edgecolor=HACKER_GREY, linewidth=2)
    axes[1].axhline(y=0.5, color=NUGGET_YELLOW, linestyle='--', linewidth=2, label='Random (50%)')
    axes[1].set_ylabel('Attack Accuracy')
    axes[1].set_title('MIA Attack Accuracy', color=HTB_GREEN, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight', facecolor=NODE_BLACK)
    plt.close()
```

Now we compute confidence values and generate the visualization:

Code: python

```python
baseline_member_conf = np.max(get_model_predictions(baseline_model, X_baseline_members, DEVICE), axis=1)
baseline_nonmember_conf = np.max(get_model_predictions(baseline_model, X_baseline_nonmembers, DEVICE), axis=1)
student_private_conf = np.max(get_model_predictions(student_model, X_private_norm, DEVICE), axis=1)
student_holdout_conf = np.max(get_model_predictions(student_model, X_holdout_norm, DEVICE), axis=1)

plot_confidence_distributions(
    baseline_member_conf, baseline_nonmember_conf,
    student_private_conf, student_holdout_conf,
    save_path="figs/pate_confidence_distributions.png"
)
```

![Side-by-side bar charts showing density difference between member and non-member prediction confidence. Left panel (Baseline Model): tall red spike at high confidence shows members receive far more confident predictions than non-members. Right panel (PATE Student): differences hover near zero across all confidence values, showing PATE eliminates the distinguishing signal.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_confidence_distributions.png)

What do these density differences reveal? In the left panel, the tall red spike at high confidence shows the baseline's vulnerability: members receive highly confident predictions far more often than non-members. This is the attack surface. In contrast, the right panel shows PATE's protection: the difference hovers near zero across all confidence values. The student model cannot distinguish private data from holdout data because it never trained on either.

## Visualizing Attack Comparison

Code: python

```python
plot_mia_comparison(
    baseline_mia_adv, student_mia_adv_private, student_mia_adv_query,
    baseline_mia_acc, student_mia_acc_private, student_mia_acc_query,
    save_path="figs/pate_mia_comparison.png"
)
```

![Dual-panel bar chart comparing membership inference attack effectiveness. Left panel (Attack Advantage): Baseline exceeds 6% threshold (red), PATE Private drops to near 0% (green), PATE Query shows expected leakage (cyan). Right panel (Attack Accuracy): Baseline at 57%, PATE Private at exactly 50% (random guessing), confirming PATE's protection.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_mia_comparison.png)

How effective is PATE's protection? The left panel's attack advantage tells the story. The baseline (red) exceeds the 5% threshold commonly used as an acceptable leakage bound. The PATE student against private data (green) drops to near zero, statistically indistinguishable from random guessing. The cyan bar shows expected leakage about query data, which is not a privacy concern since query data is public by design. The right panel confirms this interpretation: attack accuracy against private data sits precisely at 50%, the random baseline.

## Interpreting the Results

These empirical results validate PATE's theoretical privacy guarantees. The student achieves differential privacy not through noise during inference, but through the information bottleneck during training. Private data influences the student only through noisy aggregate labels, and this limited information transfer prevents meaningful membership inference.

Several observations emerge from our attack evaluation. `Baseline vulnerability scales with overfitting`: the overfitting gap in the baseline translates directly to attack success. More overfitting means more memorization, which means more distinguishable member behavior.

`PATE protection is architectural`. The student's resistance to attack does not depend on adding noise at inference time. Once trained, the student answers queries deterministically without privacy cost. This architectural guarantee is fundamentally different from techniques that require noise during inference.

`Public query leakage confirms methodology`. The leakage about query data is not a privacy concern because query data is public by assumption. This result validates our attack implementation: it successfully detects membership when the model did train on the data, while failing to detect membership for data the model never saw.

`Attack advantage reduction is substantial`. An attacker who could previously determine membership with accuracy above chance now achieves accuracy statistically indistinguishable from random guessing.

