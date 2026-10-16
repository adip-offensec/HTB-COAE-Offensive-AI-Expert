
# Evaluation and Results

After training, we evaluate your adversarially trained model against pre-generated FGSM and I-FGSM attacks, providing consistent metrics across different training runs. This section explains how to use the evaluator, interpret its output, and understand what the robustness metrics reveal about your defense effectiveness.

---

## Running the Evaluator

After training completes and saves your model, run the standalone evaluator:

  Evaluation and Results

```shell-session
@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors
```

The evaluator loads pre-generated adversarial examples from `adv_examples.safetensors`, ensuring consistent attack samples regardless of when you run evaluation. These examples were generated using a reference implementation at multiple epsilon values, preventing evaluation variance from affecting your results.

Add the `--compare` flag to see how your model improves over the vulnerable baseline:

  Evaluation and Results

```shell-session
@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors --compare
```

This loads both your model and the pre-trained baseline, running identical evaluations and displaying a side-by-side comparison.

You can also evaluate multiple models at once by passing a directory:

  Evaluation and Results

```shell-session
@htb[/htb]$ python evaluate_robustness.py --model-path models/ --compare
```

The evaluator automatically discovers model files (`.safetensors`), locates the baseline model and adversarial examples, and evaluates each model in sequence.

---

## Understanding the Output

Running the evaluator displays results for each model with a full epsilon spread table showing how robustness degrades as perturbation strength increases:

  Evaluation and Results

```shell-session
Using device: cuda
Loading adversarial examples from: adv_examples.safetensors
  500 test samples
  Epsilon spread: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Evaluating spread_epsilon_model.safetensors...

======================================================================
Model: spread_epsilon_model.safetensors
======================================================================

Clean Accuracy: 99.0% (495/500)

Epsilon          FGSM     I-FGSM  FGSM Attack  I-FGSM Attack
------------------------------------------------------------
0.10            98.4%      98.4%         1.6%           1.6%
0.20            98.0%      97.8%         2.0%           2.2%
0.30            97.6%      97.6%         2.4%           2.4% *
0.40            96.8%      97.0%         3.2%           3.0%
0.50            95.8%      95.6%         4.2%           4.4%
0.60            94.6%      94.6%         5.4%           5.4%
0.70            93.6%      93.0%         6.4%           7.0%
0.80            90.8%      89.0%         9.2%          11.0%
0.90            87.0%      84.2%        13.0%          15.8%
1.00            76.6%      77.0%        23.4%          23.0%
------------------------------------------------------------
* Primary epsilon for summary metrics

Summary (ε=0.3):
  Clean:   99.0%
  FGSM:    97.6%  (attack success:   2.4%)
  I-FGSM:  97.6%  (attack success:   2.4%)
```

Reading left to right, `FGSM` and `I-FGSM` columns show model accuracy (percentage of adversarial examples correctly classified), while `Attack` columns show how many examples fooled the model (computed as 100% minus accuracy). Lower attack values indicate stronger defense.

Notice the asterisk marking the primary epsilon (0.3) used for summary metrics. This epsilon represents a moderate perturbation strength commonly used in adversarial robustness research.

---

## Baseline Comparison

The `--compare` flag first evaluates the baseline model, then your trained models, showing improvement for each:

  Evaluation and Results

```shell-session
Evaluating baseline...

======================================================================
Model: baseline_model.safetensors
======================================================================

Clean Accuracy: 98.6% (493/500)

Epsilon          FGSM     I-FGSM  FGSM Attack  I-FGSM Attack
------------------------------------------------------------
0.10            95.2%      94.2%         4.8%           5.8%
0.20            86.6%      80.8%        13.4%          19.2%
0.30            73.8%      52.0%        26.2%          48.0% *
0.40            56.2%      26.8%        43.8%          73.2%
0.50            40.4%       9.0%        59.6%          91.0%
0.60            25.8%       2.4%        74.2%          97.6%
0.70            16.4%       0.6%        83.6%          99.4%
0.80            10.8%       0.4%        89.2%          99.6%
0.90             8.0%       0.2%        92.0%          99.8%
1.00             7.0%       0.0%        93.0%         100.0%
------------------------------------------------------------

Summary (ε=0.3):
  Clean:   98.6%
  FGSM:    73.8%  (attack success:  26.2%)
  I-FGSM:  52.0%  (attack success:  48.0%)
```

Notice the severe vulnerability pattern: while FGSM attack success rises gradually with epsilon, I-FGSM attack success climbs rapidly, reaching 91% at epsilon 0.5 and nearly 100% at higher values. This demonstrates why iterative attacks pose a greater threat than single-step attacks against undefended models.

After evaluating the baseline, the script evaluates your trained models and shows the improvement:

  Evaluation and Results

```shell-session
Evaluating spread_epsilon_model.safetensors...

======================================================================
Model: spread_epsilon_model.safetensors
======================================================================

Clean Accuracy: 99.0% (495/500)

Epsilon          FGSM     I-FGSM  FGSM Attack  I-FGSM Attack
------------------------------------------------------------
0.10            98.4%      98.4%         1.6%           1.6%
0.20            98.0%      97.8%         2.0%           2.2%
0.30            97.6%      97.6%         2.4%           2.4% *
0.40            96.8%      97.0%         3.2%           3.0%
0.50            95.8%      95.6%         4.2%           4.4%
0.60            94.6%      94.6%         5.4%           5.4%
0.70            93.6%      93.0%         6.4%           7.0%
0.80            90.8%      89.0%         9.2%          11.0%
0.90            87.0%      84.2%        13.0%          15.8%
1.00            76.6%      77.0%        23.4%          23.0%
------------------------------------------------------------

Summary (ε=0.3):
  Clean:   99.0%
  FGSM:    97.6%  (attack success:   2.4%)
  I-FGSM:  97.6%  (attack success:   2.4%)

Improvement over baseline:
  FGSM:   +23.8% (73.8% → 97.6%)
  I-FGSM: +45.6% (52.0% → 97.6%)
```

The full comparison summary shows all models side-by-side across the entire epsilon spread, from both the defender's and attacker's perspectives:

  Evaluation and Results

```shell-session
==========================================================================================
COMPARISON SUMMARY - Full Epsilon Spread
==========================================================================================

Clean Accuracy:               baseline  single_epsilon  spread_epsilon
                                 98.6%           99.4%           99.0%

FGSM Model Accuracy (defender):
------------------------------------------------------------------------------------------
Epsilon                       baseline  single_epsilon  spread_epsilon
------------------------------------------------------------------------------------------
0.10                             95.2%           99.0%           98.4%
0.20                             86.6%           98.4%           98.0%
0.30                             73.8%           97.0%           97.6% *
0.40                             56.2%           95.8%           96.8%
0.50                             40.4%           93.2%           95.8%
0.60                             25.8%           86.4%           94.6%
0.70                             16.4%           76.2%           93.6%
0.80                             10.8%           65.4%           90.8%
0.90                              8.0%           52.4%           87.0%
1.00                              7.0%           36.8%           76.6%

FGSM Attack Success (attacker):
------------------------------------------------------------------------------------------
Epsilon                       baseline  single_epsilon  spread_epsilon
------------------------------------------------------------------------------------------
0.10                              4.8%            1.0%            1.6%
0.20                             13.4%            1.6%            2.0%
0.30                             26.2%            3.0%            2.4% *
0.40                             43.8%            4.2%            3.2%
0.50                             59.6%            6.8%            4.2%
0.60                             74.2%           13.6%            5.4%
0.70                             83.6%           23.8%            6.4%
0.80                             89.2%           34.6%            9.2%
0.90                             92.0%           47.6%           13.0%
1.00                             93.0%           63.2%           23.4%

I-FGSM Model Accuracy (defender):
------------------------------------------------------------------------------------------
Epsilon                       baseline  single_epsilon  spread_epsilon
------------------------------------------------------------------------------------------
0.10                             94.2%           98.8%           98.4%
0.20                             80.8%           98.4%           97.8%
0.30                             52.0%           97.2%           97.6% *
0.40                             26.8%           95.8%           97.0%
0.50                              9.0%           92.2%           95.6%
0.60                              2.4%           84.8%           94.6%
0.70                              0.6%           71.8%           93.0%
0.80                              0.4%           55.6%           89.0%
0.90                              0.2%           41.2%           84.2%
1.00                              0.0%           24.8%           77.0%

I-FGSM Attack Success (attacker):
------------------------------------------------------------------------------------------
Epsilon                       baseline  single_epsilon  spread_epsilon
------------------------------------------------------------------------------------------
0.10                              5.8%            1.2%            1.6%
0.20                             19.2%            1.6%            2.2%
0.30                             48.0%            2.8%            2.4% *
0.40                             73.2%            4.2%            3.0%
0.50                             91.0%            7.8%            4.4%
0.60                             97.6%           15.2%            5.4%
0.70                             99.4%           28.2%            7.0%
0.80                             99.6%           44.4%           11.0%
0.90                             99.8%           58.8%           15.8%
1.00                            100.0%           75.2%           23.0%
------------------------------------------------------------------------------------------
* Primary epsilon for summary metrics

Improvement over baseline at ε=0.3:
  single_epsilon_model.safetensors: FGSM +23.2%, I-FGSM +45.2%
  spread_epsilon_model.safetensors: FGSM +23.8%, I-FGSM +45.6%
==========================================================================================
```

What patterns emerge from this comparison? Both trained models achieve similar accuracy at the primary epsilon (0.3), but their behavior diverges at higher perturbation strengths. The single-epsilon model degrades at higher epsilons because it only trained against epsilon 0.3 perturbations. In contrast, the spread-epsilon model maintains stronger robustness across the full range because it trained against varied perturbation strengths.

From the attacker's perspective, the improvement becomes especially clear. Against the baseline, I-FGSM achieves `48.0% attack success at epsilon 0.3 and nearly 100% at epsilon 0.6+`. Against the spread-epsilon model, attack success stays below 6% even at epsilon 0.6. This demonstrates robust defense across the intensity range.

---

## Interpreting Robustness Patterns

To understand your defense quality, examine the epsilon spread table closely. A well-trained model shows gradual degradation as epsilon increases, maintaining above 80% accuracy even at epsilon 1.0. This indicates robust decision boundaries that resist perturbations across the full intensity range.

Watch for these patterns in your results:

A steep drop at a specific epsilon (for example, 95% at epsilon 0.3 but 60% at epsilon 0.4) indicates `epsilon overfitting`, as discussed earlier. Solution: implement epsilon spread training.

I-FGSM accuracy 3-5% lower than FGSM is normal, since iterative attacks find stronger adversarial examples. A gap larger than 10% might indicate the defense relies on artifacts of single-step attacks.

Accuracy below 85% at epsilon 1.0 is acceptable given the extreme perturbation magnitude. At this intensity, images are visibly corrupted, and some misclassification is inevitable. Achieving 80%+ at epsilon 1.0 demonstrates strong robustness across the full range.

---

## Improving Your Results

If your model shows lower robustness than expected, several approaches can help.

#### Low Clean Accuracy

Clean accuracy significantly below the baseline (under 95%) typically indicates insufficient training on clean examples. The model has focused too heavily on adversarial robustness at the expense of basic classification ability. Solutions include reducing training epsilon, increasing the ratio of clean examples in mixed batches, or adding more epochs to allow convergence.

#### Low Robust Accuracy

Robust accuracy under 90% indicates the defense is not yet effective. Common causes include training epsilon too low (weak attacks do not teach robustness), too few epochs (insufficient time to develop robust boundaries), or implementation errors in the adversarial training loop.

Several adjustments can help: increase training epsilon to 0.25 or 0.3 for stronger adversarial examples, add more epochs (adversarial training requires more than standard training), and verify that fresh adversarial examples are generated each batch rather than reusing stale examples.

#### Analyzing Failure Cases

The `--show-failures` flag displays specific misclassified samples:

  Evaluation and Results

```shell-session
@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors --show-failures

<SNIP>
Misclassified samples (I-FGSM ε=0.3):
  Sample 1842: 3 → 8 (conf=0.87)
  Sample 2891: 7 → 1 (conf=0.72)
  Sample 4123: 4 → 9 (conf=0.65)
  Sample 5567: 9 → 4 (conf=0.91)
  Sample 7234: 2 → 7 (conf=0.58)
<SNIP>
```

Failure analysis reveals which digit pairs the model confuses under attack. The 3/8 and 4/9 pairs are common confusion points because these digits share similar features. High-confidence misclassifications (like the 9 predicted as 4 with 91% confidence) indicate the adversarial example successfully moved the input deep into the wrong class's decision region.

