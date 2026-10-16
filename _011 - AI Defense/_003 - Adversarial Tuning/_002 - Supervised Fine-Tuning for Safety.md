
# Supervised Fine-Tuning for Safety

The previous section explained how jailbreak and priming attacks exploit language model behavior. Now we turn to the defense: how do we teach a model to recognize these attacks and respond appropriately?

Supervised fine-tuning (SFT) provides the mechanism. The approach is conceptually straightforward: we show the model examples of attacks paired with appropriate refusals, and the training process adjusts model weights to increase the probability of generating similar refusals for similar inputs. Understanding how this works at a technical level reveals both the power and the limitations of adversarial tuning.

---

## The Mechanics of Supervised Fine-Tuning

How do language models generate text? By predicting the next token given all previous tokens. During training, we compare this prediction against the actual next token from the training data, and the difference drives weight updates through backpropagation.

The training objective minimizes the negative log-likelihood of the correct sequence. For a training example with tokens x1,x2,…,xn, we compute the loss as:ℒ=−∑i=1nlog⁡P(xi|x1,…,xi−1;θ)

Each term log⁡P(xi|x1,…,xi−1;θ) measures how well the model (with parameters θ) predicts token xi given the preceding context. Summing these terms across all positions gives the total loss for the sequence. Lower loss means higher probability assigned to the correct tokens.

For safety tuning, our training examples consist of jailbreak prompts followed by refusal responses. The model learns to assign high probability to refusal tokens when conditioned on jailbreak-like context. After sufficient training, this learned distribution generalizes beyond the specific examples seen during training.

---

## Low-Rank Adaptation for Efficient Training

Full fine-tuning requires updating billions of parameters and demands significant GPU memory. How can we achieve comparable results without this overhead? LoRA (Low-Rank Adaptation) provides an efficient alternative by training only a small fraction of the parameters.

The core insight behind LoRA involves the observation that weight updates during fine-tuning often lie in a low-dimensional subspace. Rather than updating the full weight matrix W∈ℝd×k directly, LoRA adds a low-rank decomposition:W′=W+BAwhere B∈ℝd×r and A∈ℝr×k with rank r≪min⁡(d,k).

During training, the original weight matrix `W` remains frozen while only `A` and `B` receive gradient updates. For our configuration with rank `r = 16` applied to attention layers, this reduces trainable parameters from roughly 1.2 billion to approximately 8 million, a reduction of 99.3%. Memory requirements drop proportionally, enabling training on consumer GPUs with 16GB or less of VRAM.

Why doesn't this dramatic parameter reduction hurt quality? Fine-tuning for specific behaviors (like refusing jailbreaks) requires adjustments that genuinely occupy a low-rank subspace of the full parameter space. We are not teaching the model entirely new capabilities but rather adjusting the decision boundaries for an existing capability (safety refusal).

---

## Target Modules and Attention Layers

Where should we attach LoRA adapters within the transformer architecture? Our configuration targets the query, key, value, and output projection matrices in attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) plus the feed-forward layers (`gate_proj`, `up_proj`, `down_proj`).

Why focus on attention layers? They control how the model weighs different parts of the input context when generating outputs. Consider a jailbreak prompt that contains both manipulation framing and a harmful request. We need the model to recognize manipulation patterns (roleplay framing, authority claims) as refusal signals rather than instructions to follow.

By adapting these attention projections, we directly influence how the model processes adversarial context. The feed-forward layers then learn to produce refusal-appropriate representations based on this modified attention pattern.

---

## Training Data Composition

Effective safety tuning requires balanced training data. A dataset containing only jailbreak refusals would teach robust refusal but might also increase refusals for legitimate requests. We need to demonstrate both when to refuse and when to help.

Our training set combines three categories: jailbreak refusals (attack prompts paired with appropriate refusal responses spanning roleplay, authority claims, hypothetical framing, and encoding tricks), priming attack defenses (examples where harmful content has been injected into the assistant prefix, teaching the model to stop), and benign conversations (legitimate queries with helpful responses covering diverse topics). The lab setup section provides the exact counts and explores the data balance in detail.

Why does the ratio matter? Safety examples should slightly outnumber benign ones, but not by too much. We determined this balance empirically: too few safety examples and the model remains vulnerable, too many and the model becomes overly cautious.

---

## The Generalization Question

A central challenge in adversarial tuning is generalization. Our training set contains specific jailbreak examples, but attackers can construct novel variations we never anticipated. How does training on known attacks help against unknown attacks?

Feature learning provides the answer. Neural networks do not memorize training examples as discrete items but rather learn distributed representations that capture patterns across examples. When we train on roleplay jailbreaks, the model does not memorize specific wordings. Instead, it learns features associated with the roleplay manipulation pattern: phrases like `you are now`, `pretend to be`, `in this game`, and `without restrictions`.

These features activate for novel jailbreaks that share structural similarity with training examples even if the exact wording differs. An attacker might phrase a roleplay jailbreak in a novel way, but if it contains the functional elements of roleplay manipulation, the learned features still activate and trigger refusal.

Empirical studies confirm this generalization. Models trained on a subset of jailbreak categories often refuse jailbreaks from held-out categories. The learned safety features capture something general about manipulation attempts rather than memorizing specific attack strings.

Generalization is not perfect, however. Sufficiently novel attack patterns may evade detection. But adversarial tuning raises the bar significantly. Attackers must now find not just any jailbreak but one that evades learned detection across multiple feature dimensions.

---

## Avoiding Over-Refusal

We must also avoid the opposite failure mode: a model that refuses too aggressively. Over-refusal occurs when safety training causes the model to decline legitimate requests. A model might refuse to discuss chemistry entirely because some chemistry requests relate to synthesizing dangerous substances. It might refuse any security question because some security questions are attack reconnaissance.

What contributes to over-refusal? Training data imbalance (where safety examples vastly outnumber benign examples) teaches the model that refusal is the common case. Overly broad refusal responses (where the model learns to refuse entire topic areas rather than specific harmful requests) create excessive caution. Insufficient benign example diversity (where benign training data does not cover topics adjacent to harmful ones) leaves ambiguous cases defaulting to refusal.

Our training approach addresses each factor. We balance the benign example count (86) against safety examples (138). Refusal responses target specific harmful requests rather than entire topic areas. Benign examples cover security, chemistry, and other sensitive topics to demonstrate that these areas have legitimate uses.

Evaluation explicitly measures helpfulness alongside safety. A model that achieves 100% jailbreak refusal but only 50% benign helpfulness has failed. Both metrics must meet their thresholds for success.

---

## From Theory to Practice

With the theoretical foundation in place, the lab sections implement this training approach step by step. The data preparation section covers loading and formatting training examples according to the model's chat template. The training section walks through model loading, LoRA adapter configuration, and executing the fine-tuning process. Each hyperparameter choice is explained in context as the training configuration is built.

After training completes, evaluation on held-out test data determines whether the model successfully learned to refuse attacks while remaining helpful. A model that achieves high training accuracy but low evaluation accuracy has overfit to specific training examples rather than learning generalizable safety patterns. Both metrics must meet their thresholds: 90% or higher on attack refusal, 85% or higher on benign helpfulness.

