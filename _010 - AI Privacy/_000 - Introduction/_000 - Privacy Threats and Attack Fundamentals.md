

# Privacy Threats and Attack Fundamentals

Machine learning models are supposed to learn patterns, not memorize individuals. Yet every model trained on personal data carries a hidden risk: it can reveal which specific individuals were in its training set. This privacy violation, known as `membership inference`, represents one of the core threats to ML systems deployed on sensitive data.

## What is a Membership Inference Attack?

A `Membership Inference Attack (MIA)` targets a straightforward question: was a specific data point used to train a given model? We provide a sample and receive a binary answer (member or non-member) based on how the model responds to that sample. Despite its simplicity, this binary classification has serious privacy implications because membership itself can reveal sensitive information.

## Privacy Attacks on Machine Learning

`MIA` belongs to a broader family of privacy attacks targeting ML systems, each extracting different types of information.

`Model inversion attacks` attempt to reconstruct training data features from model outputs. Given a model that predicts disease risk from genetic markers, an attacker might infer genetic information about training subjects. A related threat comes from `attribute inference attacks`, where we deduce sensitive attributes not directly predicted by the model. If a model predicts income and was trained on complete records, querying it strategically might reveal education levels or employment status of training members. The most direct form of leakage occurs in `training data extraction attacks`, where we recover verbatim training examples from models. Large language models that memorize specific sequences are particularly vulnerable to this approach.

Where does `MIA` fit in this landscape? It represents a basic privacy violation because it requires the least information to execute and often appears alongside other attacks. Even if attackers do not run a separate membership test first, a model that leaks membership information tends to be easier to exploit with `model inversion attacks` or `attribute inference attacks`. `MIA` success also serves as a privacy audit metric: if a model leaks membership, it likely leaks other information too. We focus on `MIA` because it establishes a baseline vulnerability that other attacks often build upon.

Consider a medical diagnosis model trained on patient records. An attacker with access only to the model's predictions could determine whether a particular patient's data was used for training. If the model was trained exclusively on cancer patients, successfully identifying someone as a training member reveals their medical status. Similar scenarios arise in financial services, where membership in a credit scoring model's training set might expose loan history, or in social platforms where presence in a content moderation dataset could imply previous violations.

This module establishes the baseline vulnerability of ML models to `membership inference` before we introduce any defenses. We implement the core attack methodology from [Shokri et al.'s 2017 paper](https://arxiv.org/abs/1610.05820), which demonstrated that standard neural networks leak substantial membership information through their prediction behavior.

## Why Membership Inference Works

`Membership inference` exploits the distinction between `memorization` and `generalization`. When we train a model, we want it to learn general patterns that apply to new data. In practice, models also memorize specific training examples, particularly those that are unusual, repeated, or near decision boundaries. This memorization creates detectable behavioral differences between how models treat data they have seen versus data they have not.

Consider what happens during gradient descent. We adjust model weights to reduce loss on training samples. After many iterations, we fit training data very closely, sometimes perfectly classifying every training example. But this tight fit does not transfer to new data. We have learned idiosyncrasies of specific training samples instead of underlying patterns. This phenomenon, commonly called `overfitting`, is the root cause of membership leakage.

Our `attack model` extracts two distinct signals from model predictions. The first is prediction confidence: how certain is the model about its prediction? Members tend to receive higher-confidence predictions because the model optimized specifically for them during training. A member might receive `[0.05, 0.95]` (95% confidence in class 1) while a non-member gets `[0.20, 0.80]` (80% confidence). The second signal is prediction correctness: does the predicted class match the true label? Models are more accurate on their training data because they have seen those exact feature combinations during training.

These signals interact in interesting ways. A high-confidence correct prediction is a strong membership indicator because both signals agree. A high-confidence incorrect prediction suggests non-membership. The challenging cases are low-confidence predictions where neither signal is decisive, and these form the bulk of attack errors. Sometimes the signals conflict: a member receives a low-confidence prediction on a genuinely ambiguous sample, or a non-member receives high confidence on a very typical sample. Our `attack model` learns to weight these signals based on how often each pattern corresponds to membership.

Model capacity also matters. Larger models with more parameters can memorize more training data, making them more vulnerable to `MIA`. A model with 10 million parameters stores more training examples in its weights than one with 100,000 parameters. Regularization techniques like dropout, weight decay, and early stopping reduce memorization but cannot eliminate it entirely. The optimization process inherently treats training data differently from unseen data, creating tension between model utility and privacy.

Our goal as attackers is to learn a binary classifier that exploits this memorization gap. Given a specific sample and its true label, we determine whether that sample was in the target model's training set. A successful attack correctly identifies training members with accuracy significantly better than random guessing (50%).

## Industry Security Frameworks

Three frameworks catalogue ML privacy risks. The `OWASP ML Security Top 10` (draft v0.3) lists `membership inference` as `ML04:2023`, alongside model inversion (`ML03:2023`) and model theft (`ML05:2023`). It assigns membership inference moderate exploitability (4/5) and moderate impact (4/5), recommending differential privacy and regularization as primary defenses.

The `OWASP Top 10 for LLM Applications` (2025) addresses generative AI with `LLM02: Sensitive Information Disclosure` covering training data leakage, and `LLM04: Data and Model Poisoning` addressing the memorization patterns that enable inference attacks.

`Google SAIF` takes an architectural approach, categorizing `membership inference` under `Sensitive Data Disclosure` and `Inferred Sensitive Data`. SAIF assigns responsibility to both model creators (implement privacy-preserving training) and model consumers (filter outputs, monitor queries).

All three frameworks converge on the same conclusion: if a model leaks membership, it likely leaks other information too. Differential privacy is the recommended countermeasure, which is why we focus on DP-SGD and PATE in subsequent sections.

