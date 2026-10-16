
# Introduction to Adversarial Training

In previous sections, you learned to defend LLM applications using `guardrails`: mechanisms that filter and validate inputs and outputs at the application layer. Guardrails operate as a perimeter defense, catching malicious prompts before they reach the model and sanitizing responses before users see them. These techniques address text-based threats like prompt injection and jailbreaking by examining what goes into and comes out of the model.

This section shifts focus from `filtering around the model` to `hardening the model itself`. Guardrails work well for discrete, text-based attacks where malicious patterns can be detected and blocked. Neural networks face a different class of threat: `adversarial perturbations` that exploit the model's internal decision boundaries. These attacks do not inject obviously malicious content. Instead, they make imperceptible changes to legitimate inputs that cause catastrophic misclassification. We cannot simply filter these inputs because they look identical to clean ones. The defense must come from within: we train the model to become inherently robust against perturbation.

Neural networks achieve remarkable accuracy on clean inputs yet collapse when confronted with carefully crafted perturbations. In previous sections, you learned to construct these `adversarial examples` using techniques like FGSM and I-FGSM, watching confident classifiers misclassify digits with perturbations invisible to the human eye. Now we flip perspectives: how do we defend against the very attacks we created?

---

## The Defense Challenge

Defending machine learning models against adversarial examples presents a distinctly different challenge than traditional security. Conventional defenses operate on discrete inputs where perturbations are obvious: a malformed packet, an unexpected character sequence, a suspicious system call. Adversarial perturbations exist in continuous space, where infinitesimally small changes to pixel values can flip predictions while remaining imperceptible. The attack surface is not a finite set of inputs to filter but an infinite manifold of near-identical variations surrounding every legitimate input.

Consider the baseline MNIST classifier we will work with in this lab. Trained using standard procedures, it achieves approximately `99% accuracy` on clean test images. Present it with the same images perturbed by FGSM at epsilon 0.3, and accuracy drops to approximately `74%`. The iterative I-FGSM attack is even more effective, reducing accuracy to around `52%`. The model has learned to recognize digits, but the features it relies upon are brittle, easily disrupted by targeted noise. This vulnerability is not a bug in implementation but a consequence of how neural networks learn decision boundaries.

![Baseline vs Robust Model Performance](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/322/adversarial_training_baseline_vs_robust.png)

---

## Why Standard Training Fails

Standard training optimizes models to minimize loss on the training distribution. The optimizer adjusts weights to correctly classify each training sample, finding decision boundaries that separate classes effectively on the data it has seen. These boundaries work well for test samples drawn from the same distribution because natural variation in the data roughly matches natural variation in the test set.

Adversarial perturbations exploit a core gap in this process. The optimizer never encounters inputs that lie slightly off the data manifold, in the spaces between natural examples. When FGSM pushes an image in the direction that maximizes loss, it moves into these unexplored regions where the model's decision boundary may curve unpredictably. A digit 3 perturbed toward the loss gradient might cross into a region the model associates with 8, not because the perturbation makes it look like an 8 to humans, but because `the model never learned` that this particular combination of pixels should still be classified as 3.

Coverage is the root cause: `standard training samples a tiny fraction of the high-dimensional input space`. Natural images concentrate on a low-dimensional manifold, and adversarial examples probe the vast uncharted territory surrounding it.

---

## The Adversarial Training Solution

What if we trained on adversarial examples directly? This simple intuition underlies adversarial training, the `most effective and widely-deployed defense against perturbation attacks`. Instead of optimizing only for clean inputs, we generate adversarial perturbations during training and force the model to classify them correctly alongside their unperturbed counterparts.

We integrate attack and defense into a single training loop. For each batch of training images, we compute FGSM perturbations using the current model weights, then train on both the original images and their adversarial versions. The model learns that a digit 3 should be recognized as 3 not only when it appears in its natural form but also when pushed in any direction the attacker might choose. Over many epochs, decision boundaries shift to accommodate these perturbations, `becoming robust to the attacks we anticipated`.

Why does this work when simply adding noise to training data does not? Random noise explores the input space uniformly, wasting capacity on regions attackers would never target. FGSM perturbations specifically target the model's vulnerabilities, the directions where loss increases most rapidly. Training against these worst-case perturbations directly addresses the weakness rather than hoping general regularization will suffice. The defense is tailored to the threat.

