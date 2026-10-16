
# Introduction to First-Order Evasion Attacks

Machine learning models appear robust when tested on normal data, often achieving 98-99% accuracy on held-out test sets. Yet these same models can be fooled by adversarial examples: inputs carefully modified to cause misclassification while remaining nearly identical to the original (that's the hope anyway). A digit image that clearly shows a "7" to human eyes can be tweaked by less than 1% and suddenly the model confidently predicts "2."

What makes these attacks possible? Neural networks learn by following gradients during training, adjusting weights to minimize loss. The same gradient information that trains the model also reveals its vulnerabilities. First-order attacks exploit this by computing gradients with respect to inputs rather than parameters, using calculus to find directions that maximally increase prediction error.

## What Are First-Order Attacks?

First-order attacks use gradient information to craft adversarial examples. During normal training, gradients tell us how to adjust model weights to reduce error. During an attack, gradients tell us how to adjust inputs to increase error.

Think of it as asking the model: "If I change this pixel slightly, how much does your prediction change?" The gradient provides that answer for every pixel simultaneously. An attacker uses those answers to make coordinated changes across all pixels, pushing the input across a decision boundary while keeping the total modification small.

These attacks work in different scenarios. In white-box attacks, the attacker has complete access to the model and can compute exact gradients. In black-box attacks, the attacker only queries the model's outputs, either estimating gradients numerically or crafting examples on a similar surrogate model and relying on transferability.

The key constraint is keeping perturbations small as possible. We measure this using norms (covered in detail in the next chapter).

## Two Fundamental Approaches

This module explores two foundational gradient-based attacks that represent different attack philosophies.

`FGSM` (`Fast Gradient Sign Method`) takes the direct approach. You decide upfront how much you're willing to change the input, then compute the gradient and move each pixel in the direction that increases loss. The "sign" part means you only look at whether each gradient component is positive or negative, not how large it is. This makes the attack extremely fast: one gradient computation, one step, done. It's simple enough to implement in a few lines of code, yet effective enough that it became the baseline for measuring adversarial robustness.

`DeepFool` asks a fundamentally different question: what's the smallest change that fools the model? Instead of choosing a budget upfront, DeepFool iteratively searches for the closest decision boundary. It approximates the boundary as a flat surface locally, takes the shortest step to that surface, then repeats with a fresh approximation. After a few iterations, it reaches the true boundary with minimal perturbation. This tells you exactly how vulnerable each input is, making it valuable for measuring model robustness quantitatively.

## Why This Matters

Adversarial examples expose a clear gap between high accuracy and true robustness. A model can correctly classify 99% of normal test images while catastrophically failing under adversarial examples. This vulnerability matters in security-critical applications.

First-order attacks are also remarkably transferable. An adversarial example crafted against one model often fools other models, even those with different architectures or training procedures. This enables realistic attacks where the adversary doesn't need full access to the target model.

For defenders, understanding these attacks is essential. They provide baselines for evaluating defensive techniques like adversarial training, input preprocessing, and detection systems. They also help identify which inputs lie close to decision boundaries and might be vulnerable to natural perturbations.

## Security Frameworks

Both of the major AI security frameworks (OWASP and SAIF) recognize evasion attacks as a major threat. [OWASP's Machine Learning Security Top 10 lists input manipulation as ML01:2023](https://academy.hackthebox.com/module/319/section/3877), the highest-ranked risk for traditional ML systems. The framework distinguishes these inference-time attacks from training-time threats like data poisoning, noting that evasion requires only the ability to query a deployed model.

[Google's Secure AI Framework](https://academy.hackthebox.com/module/319/section/3877) (SAIF) addresses evasion through defense in depth: adversarial training during development, robustness evaluation before deployment, and input filtering during operation. SAIF recommends that security teams establish red teams to continuously test production models with adversarial examples, measuring how much perturbation is needed to achieve target attack success rates.

Both frameworks emphasize that defending against evasion requires technical understanding of how these attacks work. The techniques in this module provide that foundation, enabling you to implement OWASP's recommended defenses and execute SAIF's evaluation protocols.

