
# Executing and Evaluating the Attack

We have trained our attack classifier on shadow model predictions, though it achieved only marginal accuracy (around 50%) on that data due to weak membership signals. Now comes the real test: applying this attack to the actual target model, which exhibits stronger overfitting. This chapter executes the attack, evaluates its effectiveness with multiple metrics, and interprets what the results mean for privacy.

We evaluate our attack using genuinely held-out data: the target model's training set (members) and the attack evaluation set (non-members the target never saw). This mirrors the scenario a real attacker would face: querying an unknown model to determine membership.

## Executing the Membership Inference Attack

We begin by collecting the target model's predictions on its training data (members) and on data it never trained on (non-members).

Code: python

```python
print("\n" + "=" * 60)
print("Executing Membership Inference Attack")
print("=" * 60)

preds_members = get_model_predictions(target_model, X_target_norm, DEVICE)
preds_non_members = get_model_predictions(target_model, X_attack_eval_norm, DEVICE)

print(f"\nTarget model predictions collected:")
print(f"  Members: {len(preds_members)} samples")
print(f"  Non-members: {len(preds_non_members)} samples")
```

We collect softmax probability vectors for all 24,421 target training samples (members) and all 12,210 attack evaluation samples (non-members). These represent the complete population for our attack evaluation.

#### Preparing Attack Input

We cannot concatenate member and non-member predictions before calling `prepare_attack_data()` because the function assigns membership labels based on position: the first argument gets labeled 1 (member), the second gets labeled 0 (non-member). We must call it separately for each population, passing empty arrays as placeholders for the missing counterpart.

We construct the empty array `np.zeros((0, preds_members.shape[1]))` with the correct column dimension (2 for binary classification) but zero rows, satisfying the function's input requirements without contributing any samples. Each call produces features of shape `(num_samples, 4)` with the appropriate membership label, which we'll then concatenate for evaluation.

Code: python

```python
attack_X_members, attack_y_members = prepare_attack_data(
    preds_members, np.zeros((0, preds_members.shape[1])),
    y_target, np.array([], dtype=np.int64)
)

attack_X_non_members, attack_y_non_members = prepare_attack_data(
    np.zeros((0, preds_non_members.shape[1])), preds_non_members,
    np.array([], dtype=np.int64), y_attack_eval
)

print(f"\nAttack input prepared:")
print(f"  Member features: {attack_X_members.shape}")
print(f"  Non-member features: {attack_X_non_members.shape}")
```

#### Combining Attack Data

With features prepared for both populations, we merge them into single arrays for evaluation. Note the class imbalance: we have twice as many members (24,421) as non-members (12,210), which affects metric interpretation.

Code: python

```python
attack_X_eval = np.concatenate([attack_X_members, attack_X_non_members], axis=0)
attack_y_eval = np.concatenate([attack_y_members, attack_y_non_members], axis=0)

print(f"\nTotal attack evaluation samples: {len(attack_X_eval)}")
print(f"  Members: {np.sum(attack_y_eval == 1)}")
print(f"  Non-members: {np.sum(attack_y_eval == 0)}")
```

#### Running the Attack

We feed the combined attack features through our trained `attack model` to obtain membership predictions and extract membership probabilities for ROC analysis.

Code: python

```python
attack_eval_loader = create_dataloader(attack_X_eval, attack_y_eval, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)

_, attack_predictions, attack_probs = evaluate_model(attack_model, attack_eval_loader, DEVICE)

membership_probs = attack_probs[:, 1]

print(f"\nAttack predictions generated")
print(f"  Mean membership probability: {membership_probs.mean():.4f}")
```

We set `shuffle=False` to maintain alignment with ground truth labels. The `[:, 1]` indexing extracts membership probability (class 1) from the probability array, giving confidence scores from 0.0 (definitely non-member) to 1.0 (definitely member).

## Computing Attack Metrics

Accuracy alone does not tell the full story. Different applications care about different aspects: privacy audits want high recall (find all members), while legal contexts want high precision (avoid false accusations). We compute accuracy, precision, recall, and F1 score to provide a complete picture.

These metrics use scikit-learn's scoring functions with the default 0.5 threshold on membership probabilities.

Code: python

```python
attack_accuracy = accuracy_score(attack_y_eval, attack_predictions)
attack_precision = precision_score(attack_y_eval, attack_predictions)
attack_recall = recall_score(attack_y_eval, attack_predictions)
attack_f1 = f1_score(attack_y_eval, attack_predictions)

print(f"\nMembership Inference Attack Results:")
print(f"  Attack Accuracy:  {attack_accuracy:.4f}")
print(f"  Attack Precision: {attack_precision:.4f}")
print(f"  Attack Recall:    {attack_recall:.4f}")
print(f"  Attack F1 Score:  {attack_f1:.4f}")
```

Let's interpret each metric. With `accuracy` (around 69% typical), we measure overall correct predictions. With 36,632 samples, 69% accuracy means correctly classifying about 25,276 samples. With `precision` (around 69% typical), we see what fraction of predicted members actually were members. Given the class imbalance, precision tends to be higher because most samples are members. With `recall` (around 97% typical), we measure what fraction of actual members we identified, the probability of detecting a randomly chosen member. The `F1 score` balances precision and recall, providing a single metric for comparison.

#### Attack Advantage

The most interpretable metric is the advantage over random guessing: simply subtract 0.5 from the attack accuracy. An advantage of around 0.19 (typical for this setup) means the attack performs about 19% better than random guessing, correctly identifying membership for about 69% of queries instead of 50%.

As a rough guide: advantages above 0.15 indicate high vulnerability where the model significantly leaks membership information; advantages between 0.05 and 0.15 suggest moderate vulnerability with detectable leakage; and advantages below 0.05 mean the attack performs near random. For privacy-sensitive applications, even a 5% advantage is concerning because it enables systematic exploitation at scale.

#### Storing Results

We collect all metrics and intermediate data for visualization and analysis.

Code: python

```python
results = {
    'attack_accuracy': attack_accuracy,
    'attack_precision': attack_precision,
    'attack_recall': attack_recall,
    'attack_f1': attack_f1,
    'attack_y_true': attack_y_eval,
    'attack_y_pred': attack_predictions,
    'attack_probs': membership_probs,
    'confidence_members': np.max(preds_members, axis=1),
    'confidence_non_members': np.max(preds_non_members, axis=1),
}

print("\nResults stored for visualization")
```

We store both scalar metrics and arrays for detailed analysis. The confidence arrays contain maximum prediction values for each sample, which we'll visualize to show the overfitting signal the attack exploited.

## Generating Visualizations

Visualizations help us understand why the attack succeeded and communicate results effectively. We'll create several plots that reveal different aspects of the attack.

#### ROC Curve

We visualize attack performance across all decision thresholds using the ROC curve. The area under this curve (AUC) summarizes overall attack quality: 0.5 means random guessing, 1.0 means perfect discrimination. We use `plot_attack_roc_curve()` to generate this visualization.

Code: python

```python
print("\n" + "=" * 60)
print("Generating Visualizations")
print("=" * 60)

auc_score = plot_attack_roc_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_roc.png")
)
results['attack_auc'] = auc_score

print(f"Attack AUC: {auc_score:.4f}")
```

![ROC curve for membership inference attack showing true positive rate versus false positive rate. The attack curve (green) achieves AUC of 0.568, slightly above the diagonal random guess baseline (dashed gray line at 0.5), indicating modest but consistent improvement over random guessing.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_Introduction_attack_roc.png)

Looking at the ROC curve, we see AUC of 0.568, just slightly above the 0.5 random baseline (dashed diagonal). While this seems modest, remember that the `attack model` was trained on `shadow model` data with nearly identical member/non-member distributions. Notice how the curve stays above the diagonal throughout, confirming the attack performs better than random guessing across all threshold choices.

#### PR Curve

The precision-recall curve is particularly informative for our imbalanced dataset (2:1 member to non-member ratio). Unlike ROC where the baseline is 0.5, the PR baseline is the fraction of positive samples (0.667).

Code: python

```python
plot_precision_recall_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_pr.png")
)
```

![Precision-recall curve for membership inference attack. The attack curve (green) achieves average precision of 0.710, staying above the 0.667 baseline (dashed gray line) across most recall values, demonstrating the attack learns real membership signals.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_Introduction_attack_pr.png)

Our PR curve achieves average precision of 0.710, above the 0.667 baseline. Notice how precision stays above 0.7 for most recall values, meaning 70% of samples we predict as members actually are members. This modest but consistent improvement over the baseline demonstrates the attack learns real membership signals.

#### Confidence Distribution

The confidence distribution visualization reveals the underlying signal our attack exploited: the difference in `prediction confidence` between members and non-members.

Code: python

```python
plot_confidence_distributions(
    results['confidence_members'],
    results['confidence_non_members'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}confidence_distributions.png")
)

print(f"\nMean confidence - Members: {np.mean(results['confidence_members']):.4f}")
print(f"Mean confidence - Non-Members: {np.mean(results['confidence_non_members']):.4f}")
```

![Histogram comparing prediction confidence distributions for members versus non-members on the target model. Members (green) have mean confidence of 0.930; non-members (red) have mean confidence of 0.923. Both distributions cluster heavily at high confidence (0.95+) with substantial overlap.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_Introduction_confidence_distributions.png)

Here we see the core signal our attack exploits: a small but consistent gap where members have mean confidence 0.9301 while non-members have 0.9226. This 0.75% difference, while subtle, persists across tens of thousands of samples. Both distributions cluster heavily at high confidence (0.95+), explaining why the attack must rely on statistical patterns instead of easily separating the populations. The substantial overlap in the middle range causes the attack errors.

#### Attack Metrics

Finally, a bar chart summarizes all attack metrics in one view, making it easy to compare performance across measures and against the 0.5 random baseline.

Code: python

```python
plot_attack_accuracy_comparison(
    results,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_metrics.png")
)
```

![Bar chart summarizing membership inference attack performance metrics. Accuracy: 69.2% (green), AUC: 0.568 (blue), Precision: 69.0% (orange), Recall: 97.6% (cyan). A dashed red line marks the 50% random guess baseline. All metrics exceed baseline, with recall notably high.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_Introduction_attack_metrics.png)

Our attack achieves 69.15% accuracy (19.15% advantage over random), 68.99% precision, and notably high 97.58% recall. The high recall indicates the attack aggressively predicts membership, catching most true members but also generating false positives among non-members. Notice that AUC at 0.5675 appears lower because it measures discrimination across all thresholds, while the other metrics use a fixed 0.5 threshold that happens to work well for this class-imbalanced dataset.

## Saving Results

We save all results to a JSON file for future reference and comparison. The output dictionary has three sections: target model performance, attack results, and configuration.

Code: python

```python
output = {
    'target_model': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'overfitting_gap': float(train_acc - test_acc),
    },
    'attack_results': {
        'accuracy': float(results['attack_accuracy']),
        'precision': float(results['attack_precision']),
        'recall': float(results['attack_recall']),
        'f1_score': float(results['attack_f1']),
        'auc': float(results['attack_auc']),
        'advantage': float(results['attack_accuracy'] - 0.5),
    },
    'configuration': {
        'random_seed': RANDOM_SEED,
        'num_shadow_models': SHADOW_MODEL_CONFIG['num_shadow_models'],
        'target_architecture': TARGET_MODEL_CONFIG['hidden_layers'],
        'attack_architecture': ATTACK_MODEL_CONFIG['hidden_layers'],
    }
}
```

We wrap NumPy floats with `float()` to ensure JSON compatibility. Now write the dictionary to disk:

Code: python

```python
results_path = os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_results.json")
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {results_path}")
```

The three sections capture target model performance (showing the overfitting gap), attack results (all computed metrics), and configuration (for reproducibility).

## Interpreting the Results

Running the complete pipeline produces output like this:

Code: txt

```txt
Target Model Performance:
  Training Accuracy: 0.9360
  Test Accuracy:     0.8247
  Overfitting Gap:   0.1113

Membership Inference Attack Results:
  Attack Accuracy:  0.6915
  Attack Precision: 0.6899
  Attack Recall:    0.9758
  Attack F1 Score:  0.8083
  Attack AUC:       0.5675

Confidence Analysis:
  Mean confidence (members):     0.9301
  Mean confidence (non-members): 0.9226
  Confidence gap:                0.0075
```

What do these numbers mean in practice? Consider a healthcare organization whose model was trained on 24,000 patient records. An attacker with API access can now determine, with 69% accuracy, whether any specific individual's data was used for training. Given 1,000 queries, random guessing yields 500 correct answers. Our attack gets 692, which translates to 192 additional patients whose presence in the training set is revealed. For sensitive domains (medical records, financial data, criminal justice), this systematic leakage enables targeted identification at scale.

Notice the relationship between metrics. High recall (97.58%) with moderate precision (68.99%) means the attack aggressively labels samples as members. It catches nearly all true members but includes false positives. This bias emerges from the 2:1 class imbalance and the 0.5 decision threshold. A privacy auditor might prefer this configuration (find all potential leaks), while a legal context might demand higher precision (avoid false accusations).

Why does AUC (0.5675) appear so much lower than accuracy (69.15%)? AUC measures discrimination across all possible thresholds, while accuracy uses a fixed 0.5 cutoff. Our class-imbalanced dataset happens to work well with that default threshold. At other thresholds, performance degrades, which the modest AUC reflects. This gap between metrics is characteristic of membership inference attacks on imbalanced data.

