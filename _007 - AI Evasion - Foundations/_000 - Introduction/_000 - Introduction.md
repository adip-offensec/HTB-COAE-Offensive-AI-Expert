
# Introduction to AI Evasion Attacks

Evasion attacks manipulate inference-time inputs to cause a trained model to produce incorrect outputs. Adversarial machine learning, the discipline that studies these hostile interactions with machine learning systems, treats this interference during the inference phase as a distinct threat because it bypasses safeguards built into the training pipeline.

Understanding evasion is therefore essential for anyone securing deployed AI models, such as AI Red Teamers.

## The AI Attack Landscape

To understand evasion, place it within the lifecycle of adversarial machine learning. AI systems can be attacked during training or at inference, and the distinction drives very different defenses.

Training‑time attacks change what the model learns. `Data poisoning` injects or alters training samples so the model internalizes biased patterns. `Label manipulation` tampers with annotations so ground truth no longer matches reality. `Trojan attacks` implant hidden triggers that activate specific behavior when an attacker’s pattern appears in the input.

These `training-time attacks` differ from `inference-time manipulation`, which changes only what the model sees when making a prediction. Data poisoning requires access to the dataset or pipeline and tends to shift behavior globally. Evasion, by contrast, leaves training and parameters untouched and succeeds by sending crafted inputs through the normal interface so a single example crosses the model’s learned decision boundary.

A property called transferability links these settings to real deployments. An adversarial example crafted against a surrogate model can often fool different production models, particularly when architectures or data are similar. This enables offline preparation and black-box attacks where the defender exposes only an API but not internals.

## Evasion Attacks in Traditional ML vs. LLMs

Evasion in traditional machine learning targets fixed feature representations, so small, controlled edits can push an example across a learned `decision boundary`. In a spam filter that uses a `bag-of-words` model, adding benign tokens changes term frequencies and therefore the posterior used for classification, while the message still reads the same to a human. In a static malware detector, rearranging sections or perturbing imports alters byte patterns and crafted signatures without changing program intent. The common thread is direct manipulation of the summarized features the model consumes, with the goal of moving a single prediction to the other side of its threshold at inference time.

By contrast, large language models produce open-ended text and follow instructions, which makes `prompt injection` the central evasion pattern. An attacker places adversarial directives into the prompt or surrounding context so the model prioritizes those instructions over the original `system` guidance, all without changing parameters. The effect is still inference-time only, yet the consequences differ because the output is itself an action surface, for example a block of code, a database query, or markup that another system might execute or render. When tool routing or API calling wraps the model, those outputs can trigger follow-on behavior that extends the attack beyond the initial response.

Both settings share the same core idea, modify only the input seen at prediction time and steer the model’s behavior. The main difference lies in where the leverage comes from. Traditional models expose structured features and a label threshold, so evasion edits target the statistics that feed the classifier. LLMs expose conversational state and instruction following, so evasion edits target the model’s decision process through carefully placed text and context management. This distinction explains why we begin with structured classifiers, the mechanics are easy to observe and measure, then we map the intuition to LLM prompts where the surface is richer but the objective is the same.

## This Module's Focus

This module anchors these concepts through a hands-on technique: the `GoodWords attack`. This attack manipulates probabilistic spam filters by inserting carefully selected benign tokens. Because Naive Bayes models assume conditional independence between features, adding those tokens shifts posterior probabilities enough to force a misclassification without raising obvious suspicion.

Studying this method builds intuition that spans domains. The attack exploits model assumptions, highlights the difference between white-box and black-box knowledge, and underscores the trade-off between subtle perturbations and reliable success, which also generalize well to other classification tasks and highlight a fundamental approach to adversarial manipulation.

