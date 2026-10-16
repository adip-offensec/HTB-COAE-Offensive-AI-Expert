
# Analyzing the Privacy-Utility Tradeoff

With the baseline and both DP-SGD models trained, we can now compare their performance directly. This section generates a summary table, creates visualizations, and interprets what the numbers reveal about the privacy-utility tradeoff.

## Helper Functions

Before analyzing results, we need two helper functions. The first formats a comparison table for terminal display, useful for immediate feedback during experimentation. The second exports metrics to JSON for automated pipelines or later analysis.

Code: python

```python
def print_comparison_table(test_acc_baseline, test_acc_dp10, test_acc_dp3,
                           mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
                           final_epsilon_10, final_epsilon_3):
    """Print a comparison table of all model results."""
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Test Acc':<12} {'MIA Adv':<12} {'Epsilon':<10}")
    print("-" * 55)
    print(f"{'Baseline':<20} {f'{test_acc_baseline:.2f}%':<12} {mia_adv_baseline:<12.4f} {'∞':<10}")
    print(f"{'DP (ε=10)':<20} {f'{test_acc_dp10:.2f}%':<12} {mia_adv_dp10:<12.4f} {final_epsilon_10:<10.2f}")
    print(f"{'DP (ε=3)':<20} {f'{test_acc_dp3:.2f}%':<12} {mia_adv_dp3:<12.4f} {final_epsilon_3:<10.2f}")
```

The baseline row displays `∞` for epsilon since an unprotected model has infinite privacy loss.

The second helper function exports all metrics to JSON for programmatic access:

Code: python

```python
def save_results(test_acc_baseline, test_acc_dp10, test_acc_dp3,
                 mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
                 final_epsilon_10, final_epsilon_3, save_path="output/results.json"):
    """Save all results to a JSON file."""
    results = {
        "baseline": {
            "test_accuracy": float(test_acc_baseline),
            "mia_advantage": float(mia_adv_baseline),
        },
        "dp_eps10": {
            "test_accuracy": float(test_acc_dp10),
            "mia_advantage": float(mia_adv_dp10),
            "epsilon": float(final_epsilon_10),
        },
        "dp_eps3": {
            "test_accuracy": float(test_acc_dp3),
            "mia_advantage": float(mia_adv_dp3),
            "epsilon": float(final_epsilon_3),
        },
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
```

We wrap each metric in `float()` to ensure JSON serialization works correctly (PyTorch tensors would otherwise fail to serialize). Structuring the JSON with nested dictionaries (`baseline`, `dp_eps10`, `dp_eps3`) makes it easy to load and compare specific model results programmatically.

## Comparing All Three Models

With all three models trained, the comparison table reveals exactly how privacy strength trades off against accuracy:

Code: python

```python
print_comparison_table(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3
)
```

Reading across the rows, accuracy declines as privacy strengthens (67% → 58% → 53%). Reading the `MIA` advantage column, privacy leakage decreases (0.019 → 0.008 → 0.004), with each step cutting the advantage by roughly 50-60%. Both DP-SGD models reduce membership inference advantage relative to the baseline, but the lower-epsilon model offers stronger privacy at the cost of larger accuracy drop.

## Generating Visualizations

Numbers tell part of the story. Visualizations make the tradeoff immediately intuitive.

Code: python

```python
plot_accuracy_comparison(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    save_path="figs/dp_sgd_accuracy_comparison.png"
)
print("Saved accuracy comparison to figs/dp_sgd_accuracy_comparison.png")
```

![Bar chart comparing test accuracy across three models. Baseline (no DP) achieves 64.2% in green, DP-SGD with ε=10 achieves 58.5% in blue, and DP-SGD with ε=3 achieves 52.3% in red. The decreasing bars show how stronger privacy protection reduces model accuracy.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_dp_sgd_accuracy_comparison.png)

Color reinforces the tradeoff: green for the high-utility baseline, blue for moderate privacy (ε=10), and red for strong privacy (ε=3). The green baseline bar reaches approximately 64%, the blue ε=10 bar sits at 58%, and the red ε=3 bar drops to 52%. The visual gap between bars immediately conveys the privacy cost.

Now create the dual-axis privacy-utility tradeoff plot:

Code: python

```python
plot_privacy_utility_tradeoff(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3,
    save_path="figs/dp_sgd_privacy_utility_tradeoff.png"
)
print("Saved privacy-utility tradeoff to figs/dp_sgd_privacy_utility_tradeoff.png")
```

![Dual-panel line chart showing privacy-utility tradeoff. Left panel: test accuracy decreases from 64% to 52% as privacy budget (epsilon) decreases from infinity to 3. Right panel: membership inference advantage drops from 0.022 to near 0.004 as epsilon decreases. Both panels use inverted x-axis with stronger privacy on the right.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_dp_sgd_privacy_utility_tradeoff.png)

The left panel plots test accuracy versus privacy budget, with an inverted x-axis placing stronger privacy (lower epsilon) on the right. The right panel shows `MIA` advantage versus privacy budget. Both curves slope downward left-to-right as epsilon increases, confirming that weaker privacy (higher epsilon) yields both better accuracy and higher vulnerability. The visual symmetry between panels reinforces how accuracy and privacy move together.

## Saving Results

For future reference or programmatic analysis, we persist all metrics to JSON:

Code: python

```python
save_results(
    test_acc_baseline, test_acc_dp10, test_acc_dp3,
    mia_adv_baseline, mia_adv_dp10, mia_adv_dp3,
    final_epsilon_10, final_epsilon_3,
    save_path="output/results.json"
)
print("Results saved to output/results.json")
```

## Interpreting the Results

Several patterns emerge from these comparisons, each showing how DP-SGD performs as a membership inference defense.

Privacy protection scales with noise. Moving from baseline to ε=10 reduced membership advantage by roughly 57% (from 0.019 to 0.008), and moving to ε=3 cut it roughly in half again (to 0.004). Each reduction in epsilon corresponds to more noise and proportionally less membership leakage. The utility cost, while substantial, remains bounded: the baseline achieved 67% test accuracy, dropping to 58% at ε=10 (9 percentage points) and 53% at ε=3 (14 percentage points total). The model remains useful for classification even under strong privacy constraints.

The overfitting gap tells a complementary story. The baseline had a 10% gap between training and test accuracy, indicating memorization of training-specific patterns. DP-SGD at ε=10 reduced this to 3%, and ε=3 reduced it to just 1%. Noise prevents the model from fitting individual examples too closely, which is exactly what makes membership inference harder.

Diminishing returns emerge at lower epsilon values. Going from ε=∞ (baseline) to ε=10 cost 9 percentage points of accuracy for 0.011 advantage reduction (roughly 1.2 points per 0.001 advantage). Going from ε=10 to ε=3 cost 5 percentage points for only 0.004 additional advantage reduction (1.25 points per 0.001 advantage). The marginal cost of privacy stays roughly constant, but the marginal benefit shrinks. At some point, the model approaches random-guess `MIA` advantage (0.0), and further noise provides no additional privacy benefit while continuing to cost accuracy.

## Limitations and Practical Considerations

DP-SGD provides strong theoretical guarantees, but understanding its limitations helps determine when it is the right choice. Computational overhead (2-5x training time) may be prohibitive for large models. Utility loss at strong privacy levels (ε < 1) can exceed 20 percentage points. The protection scope covers only individual sample influence, not aggregate patterns or non-privacy attacks.

#### Computational Overhead

Expect training to take 2-5x longer due to per-sample gradient computation. Memory usage also scales with batch size because we store individual gradients before clipping. For large models or limited hardware, this overhead can be prohibitive. Techniques like gradient accumulation and mixed-precision training can help, but DP-SGD remains more expensive than non-private training.

#### Utility Loss at Strong Privacy

Achieving very strong privacy (ε<1) while maintaining useful accuracy is challenging. Our ε=3 model lost 14 percentage points compared to the baseline. Reaching ε=1 might require losing 20-30 percentage points or more. For some applications, this utility loss is unacceptable. Research continues on techniques to improve the privacy-utility tradeoff, including better architectures, pre-training on public data, and advanced optimization methods.

#### What DP-SGD Protects Against

The noise bounds each sample's influence on final parameters, defending against membership inference and limiting what attackers can learn about individual training records. Aggregate patterns remain learnable, however. An attacker could still infer statistical properties of the training data (average age, common features) even from a DP-trained model.

Some attacks fall outside this protection entirely. Model functionality stealing (training a surrogate model from API queries) remains possible since it doesn't depend on individual sample influence. Adversarial example attacks are unaffected. The protection is specifically about privacy of training data, not security of the deployed model.

#### Alternative Approaches

Other approaches achieve privacy differently, each with distinct tradeoffs.

`PATE` (Private Aggregation of Teacher Ensembles), covered in the next section, uses an ensemble of teachers trained on disjoint data partitions. Privacy comes from noisy voting rather than noisy gradients. The student model learns from aggregated teacher predictions, never seeing the original training data. This approach often achieves better utility than DP-SGD at equivalent privacy levels but requires more computational resources (training multiple teachers) and works best when you can partition data cleanly.

`Local differential privacy` adds noise at data collection time, before training even begins. Each user perturbs their own data locally, so even the data curator never sees true values. This provides stronger trust guarantees (you don't need to trust the model trainer) but typically requires much more data for equivalent utility since noise compounds across the entire dataset.

`Federated learning` keeps raw data on user devices entirely, training local models that share only gradient updates with a central server. Privacy comes from data minimization rather than noise injection. Combined with secure aggregation, federated learning can provide strong privacy guarantees, but it introduces communication overhead and struggles with non-IID data distributions across devices.

Choosing among these depends on your specific threat model (who do you need to protect against?), data distribution (can you partition cleanly? is data IID?), and utility requirements (how much accuracy can you sacrifice?).

## Demonstration Complete

Code: python

```python
print("\n" + "=" * 80)
print("  DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nFindings:")
print(f"  Baseline MIA advantage: {mia_adv_baseline:.4f}")
print(f"  DP (ε=10) MIA advantage: {mia_adv_dp10:.4f} ({improvement_10:+.4f})")
print(f"  DP (ε=3) MIA advantage: {mia_adv_dp3:.4f} ({improvement_3:+.4f})")
print("=" * 80)
```

