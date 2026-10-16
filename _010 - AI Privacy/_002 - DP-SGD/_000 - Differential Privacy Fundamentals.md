
# Differential Privacy Fundamentals

We've seen how membership inference attacks exploit `overfitting` to recover information about individual training samples. Those attacks rely on a simple but powerful signal: models tend to be more confident on points they have seen during training than on points they have only seen at test time. Instead of defending at the model output level, we can attack the root cause during training itself.

Instead of redesigning the model architecture or hiding internal logits, DP-SGD modifies the training procedure itself. Think of it as controlled amnesia. During optimization we clip every `per-sample gradient` to a fixed L2 norm bound and then add carefully calibrated random noise before each parameter update. Over many training steps, those noisy, clipped gradients ensure that the model’s final parameters do not depend too strongly on any single training example.

We use the `CIFAR-10` image classification dataset as our running example. CIFAR-10 contains fifty thousand training images and ten thousand test images spread across ten classes such as `airplane`, `cat`, and `ship`. We first train a non-private convolutional neural network on this dataset, evaluate how vulnerable it is to a confidence-based membership inference attack, and then retrain the same architecture using DP-SGD with different privacy budgets.

## Differential Privacy: Formal Definition

A randomized algorithm M satisfies (ε,δ)-differential privacy if for any two datasets D and D′ that differ in exactly one record, and for any subset of outputs S:

P[M(D)∈S]≤eε⋅P[M(D′)∈S]+δ

This inequality bounds how much adding or removing a single person’s data can change the algorithm’s output distribution. The term eε acts as a multiplicative bound: if ε=1, outputs can be at most e≈2.7 times more likely with your data than without. The δ term allows a small probability of complete failure, where the guarantee does not hold at all.

Why does this definition matter for machine learning? Consider an attacker who observes a trained model and tries to determine whether Alice's record was in the training set. Without DP, the model might behave very differently with and without Alice's data, revealing her membership with high confidence. With DP, the model's behavior is nearly identical regardless of Alice's presence, limiting what any attacker can learn.

We control privacy strength through ε, where smaller values mean stronger privacy. At ε=1, the guarantee is very strong, providing near-complete indifference to any one person’s data. At ε=3, privacy remains strong with individual influence heavily suppressed. At ε=10, privacy is modest and the model might reveal aggregate patterns.

The parameter δ captures the probability that the guarantee might fail, and it is usually set to a very small value relative to the dataset size. For `CIFAR-10`’s fifty thousand training examples, setting δ=10−5 means the privacy guarantee holds with probability 0.99999.

## How DP-SGD Works

In standard stochastic gradient descent, we sample a batch of examples, compute the gradient of the loss with respect to each parameter, average these gradients over the batch, and then step the parameters in the negative gradient direction. Baseline SGD has perfect information: every per-sample gradient flows directly into parameter updates, unclipped and uncorrupted. DP-SGD deliberately corrupts this process through three mechanisms: `gradient clipping` bounds each sample's influence, `noise addition` obscures individual contributions, and `privacy composition` tracks cumulative budget across training steps.

#### Gradient Clipping and Sensitivity

The key to adding privacy-preserving noise is knowing how much any single sample can affect the computation. This quantity is called `sensitivity`. For a function f, sensitivity measures the maximum change in f’s output when we add or remove one record from the input dataset. If sensitivity is unbounded, we cannot calibrate noise appropriately.

In standard SGD, sensitivity is effectively infinite. A single outlier sample could produce an enormous gradient that dominates the entire batch update. DP-SGD solves this by clipping each per-sample gradient so that its L2 norm never exceeds a fixed constant `max_grad_norm`. If a sample produces a gradient with norm 5.2 and `max_grad_norm = 1.0`, DP-SGD scales that gradient down by a factor of 5.2, reducing its norm to exactly 1.0.

After clipping, the sensitivity of the gradient sum is exactly `max_grad_norm`. Adding or removing any single sample can change the sum by at most one clipped gradient, which has norm at most `max_grad_norm`. This bounded sensitivity is what makes noise calibration possible.

Clipping discards gradient information from samples with large gradients, which are often the most informative for learning. Aggressive clipping (small `max_grad_norm`) provides stronger privacy but loses more gradient signal, slowing convergence. Choosing the clipping threshold requires balancing privacy against learning efficiency.

#### Noise Addition and the Gaussian Mechanism

With sensitivity bounded, we can add calibrated noise using the `Gaussian mechanism`. For a function f with L2 sensitivity Δf, adding Gaussian noise with standard deviation σ=Δf⋅2ln⁡(1.25/δ)/ε achieves (ε,δ)-differential privacy.

In DP-SGD, we add zero-mean Gaussian noise to the sum of clipped gradients before averaging. The noise standard deviation is σ=max_grad_norm×noise_multiplier, where the noise multiplier depends on the target ε, δ, and number of training steps.

For ε=10 with batch size 256 and `max_grad_norm = 1.0` in our CIFAR-10 configuration, Opacus typically chooses a `noise multiplier` around 1.2, meaning we add Gaussian noise with standard deviation 1.2 to each clipped gradient sum. For ε=3, the same setup usually requires a `noise multiplier` around 3.8, injecting much more randomness and making learning harder. These values come from the `privacy accountant` for this specific dataset, batch size, and epoch count rather than from the closed-form Gaussian mechanism alone.

#### Privacy Accumulates Across Training

A single noisy gradient step provides strong privacy. But we take thousands of gradient steps during training, and each step reveals a little more about the training data. Privacy composes across steps: the total privacy loss grows with the number of updates.

Naive composition would add epsilon values: 1000 steps at ε=0.01 each would yield total ε=10. Advanced composition theorems provide tighter bounds, and `Rényi Differential Privacy` (`RDP`) accounting (used by Opacus) achieves even tighter tracking. We let Opacus handle this accounting automatically through its `PrivacyEngine`, which computes the `noise multiplier` needed to achieve a target ε after a specified number of epochs.

Why does more training require more noise per step? To maintain the same final privacy budget across more steps, each individual step must leak less. This means higher `noise multiplier`, which makes optimization harder. The privacy-utility tradeoff is not just about final epsilon but also about how that budget is spent across training.

## Choosing Privacy Budgets

We’ll train DP-SGD models at two epsilon values: ε=10 and ε=3. These represent points on the privacy-utility spectrum that demonstrate the tradeoff effectively.

ε=10 represents a modest privacy guarantee that preserves most model utility. In practice, many machine learning applications use epsilon values in the 1-10 range. Apple’s differential privacy implementations use epsilon values around 2-8 for various features. Research benchmarks often use ε=8 or ε=10 as a "reasonable privacy" baseline. At this level, the model is provably more private than no protection, but the guarantee allows some membership inference success.

ε=3 represents stronger privacy with more noticeable utility impact. This level provides meaningful protection against membership inference while still producing a useful classifier. Academic research often targets ε=1-3 for "strong privacy" demonstrations. The increased noise makes optimization harder, resulting in lower accuracy, but the privacy guarantee is substantially stronger.

Why not ε=1 (very strong privacy)? Achieving ε=1 on CIFAR-10 with reasonable accuracy requires careful architecture design, longer training with smaller learning rates, and often pre-training on public data. For a demonstration of the basic tradeoff, ε=3 shows the effect of stronger privacy without requiring advanced techniques.

## Theoretical Guarantees vs Empirical Measurement

Why measure `MIA` empirically when differential privacy provides theoretical guarantees? The two perspectives are complementary and reveal different aspects of privacy.

Differential privacy provides a `worst-case` bound. The guarantee holds against any possible attack, including attacks we have not yet imagined. An (ε,δ)-DP mechanism bounds the success of the optimal adversary by limiting how much any single record can change the distribution of outputs. If ε=1, the likelihood of any particular outcome can increase by at most a factor of e≈2.7 when one record is added or removed. Mapping this likelihood bound to concrete membership advantage requires additional assumptions, so it is best viewed as an upper bound on how much any attack can improve over random guessing, regardless of the adversary’s computational power or auxiliary information.

Empirical `MIA` measures `average-case` vulnerability using a specific attack strategy. A confidence-threshold attack is simple and does not exploit all available information. A more sophisticated attack (like the shadow model approach from the previous section) might achieve higher accuracy. The empirical advantage we measure is a lower bound on true vulnerability, while the DP guarantee is an upper bound.

Why is empirical `MIA` often much lower than the theoretical bound? Several factors contribute. The theoretical bound assumes a worst-case sample that maximally affects the model, while most samples have modest influence. The bound also assumes an optimal adversary with unlimited computational resources, while our attack uses a simple threshold. Finally, the bound holds uniformly for all samples, while some samples are much more vulnerable than others.

For privacy auditing, both perspectives matter. Empirical `MIA` tells us how vulnerable the model is to known attacks today. The DP guarantee tells us how vulnerable it could be to any attack, including future ones. A model with low empirical `MIA` but no DP guarantee might be broken by a better attack tomorrow. A model with a DP guarantee provides protection regardless of attack improvements.

