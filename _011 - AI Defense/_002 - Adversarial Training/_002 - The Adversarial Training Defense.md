
# The Adversarial Training Defense

Adversarial training addresses the root cause of model vulnerability: standard training never shows the model what to do with perturbed inputs. The defense is conceptually simple yet highly effective. We generate adversarial examples during training and force the model to classify them correctly alongside clean inputs. This section explains the mechanics, explores the mathematical foundation, and introduces epsilon spread training for comprehensive robustness.

---

## Core Concept: Vaccinating the Model

The defense operates on a vaccination principle. `Expose the model to weakened versions of attacks during training so it develops immunity before deployment`. Just as medical vaccines train the immune system to recognize pathogens, adversarial training teaches neural networks to recognize perturbed inputs.

Each training batch undergoes a two-phase process. First, we compute adversarial perturbations using the current model's gradients, generating worst-case examples that the model would currently misclassify. Second, we train on both the clean examples and their adversarial counterparts, updating weights to correctly classify both.

The iterative nature matters. Early in training, the model is weak and generates weak adversarial examples. As training progresses, the model improves and so do its adversarial examples. Late-stage adversarial examples reflect a strong model's vulnerabilities, which are precisely the vulnerabilities that matter at deployment. This `co-evolution` ensures the defense tracks the threat.

---

## The Training Loop

The adversarial training loop modifies standard training by inserting attack generation between data loading and gradient updates. We illustrate with pseudocode before diving into implementation details:

Code: pseudocode

```pseudocode
for each batch (images, labels):
    1. Generate adversarial examples: adv_images = FGSM(model, images, labels, epsilon)
    2. Combine batches: combined = concat(images, adv_images)
    3. Duplicate labels: combined_labels = concat(labels, labels)
    4. Forward pass: outputs = model(combined)
    5. Compute loss: loss = cross_entropy(outputs, combined_labels)
    6. Backward pass and update: loss.backward(), optimizer.step()
```

`Step 1` generates fresh adversarial examples using the current model state. These examples target the model's current weaknesses, not some fixed attack computed before training. The attack adapts as the model adapts.

`Steps 2 and 3` create the combined training batch. Concatenating clean and adversarial images doubles the effective batch size. Both versions share the same true labels because adversarial examples should be classified as their original class, not as whatever the attack causes the model to predict.

`Steps 4-6` perform standard backpropagation on the combined batch. The loss function treats clean and adversarial examples equally, penalizing misclassification of both. Gradients flow through both types of examples, pushing weights toward correct classification on the full distribution.

---

## Mathematical Foundation

Adversarial training solves a min-max optimization problem. Let θ represent model parameters, L the loss function, and ℬϵ(x) the set of all inputs within epsilon of x in infinity norm. Standard training minimizes expected loss over the data distribution:

min⁡θ𝔼(x,y)∼D[L(fθ(x),y)]

Adversarial training instead minimizes the worst-case loss within the epsilon ball:

min⁡θ𝔼(x,y)∼D[max⁡x′∈ℬϵ(x)L(fθ(x′),y)]

The inner maximization finds the worst-case perturbation for the current model. The outer minimization adjusts model weights to reduce this worst-case loss. FGSM provides an efficient approximation to the inner maximization: rather than searching exhaustively over ℬϵ(x), we take a single gradient step to find an approximately worst-case input.

This formulation explains why adversarial training produces robust decision boundaries. Minimizing worst-case loss within an epsilon ball effectively pushes decision boundaries away from each training point by at least epsilon. Points that were originally near a boundary get moved inside the margin, and the boundary must accommodate this expanded margin without misclassifying other points. The resulting boundaries maintain separation even under perturbation.

---

## Balancing Clean and Adversarial Accuracy

Adversarial training involves a tradeoff. Making the model robust to perturbations typically reduces accuracy on clean inputs. The model allocates capacity to handle variations it would not encounter naturally, leaving less capacity for fine-grained distinctions on clean data.

The tradeoff manifests in practice. Recall from the introduction that a standard MNIST classifier achieves approximately 99% clean accuracy but collapses to around 5% under FGSM attack. `The same architecture with adversarial training might achieve 98% clean accuracy but 95% robust accuracy`. The modest drop in clean accuracy buys a 90 percentage point improvement in robust accuracy.

We can tune this tradeoff by adjusting the ratio of clean to adversarial examples in training. A 50/50 mix (our default) emphasizes robustness. A 70/30 mix preserving more clean examples might achieve 98.5% clean and 90% robust. `The appropriate ratio depends on deployment requirements`: systems facing constant adversarial probing should favor robustness, while systems expecting mostly clean inputs might favor accuracy.

---

## Epsilon Spread Training

The previous section described `epsilon overfitting`: models trained at a single epsilon value may fail at other perturbation magnitudes. Epsilon spread training addresses this vulnerability by sampling random epsilon values during training. Instead of always using epsilon 0.3, we sample uniformly from a range like `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`. Each batch encounters a different perturbation magnitude, teaching the model to maintain correct classifications across the full range:

Code: pseudocode

```pseudocode
for each batch (images, labels):
    1. Sample random epsilon from spread: batch_epsilon = random_choice([0.1, 0.2, ..., 1.0])
    2. Generate adversarial examples: adv_images = FGSM(model, images, labels, batch_epsilon)
    3. Continue with combined training as before...
```

The random sampling exposes the model to the full spectrum of attack intensities. Low-epsilon batches teach the model to maintain precision on subtle perturbations. `High-epsilon batches teach robustness against severe attacks`. The model learns that inputs might be perturbed by any amount up to 1.0 and must classify all of them correctly.

Empirical results demonstrate the value of epsilon spread training. Models trained only at epsilon 0.3 achieve high accuracy at that specific value but degrade significantly at higher epsilons:

|Epsilon|Single-Epsilon Training|Spread Training|
|---|---|---|
|0.3|98%|98%|
|0.5|91%|95%|
|0.7|75%|87%|
|1.0|33%|66%|

Both approaches achieve similar accuracy at the training epsilon (0.3), but spread training maintains robustness across the full range while single-epsilon training collapses at higher values.

---

## Training Hyperparameters

Several hyperparameters affect adversarial training effectiveness. We typically use a lower learning rate than for standard training (`0.001` vs `0.01`) because the combined loss landscape from clean and adversarial examples is more complex. Weight decay (L2 regularization) helps prevent overfitting to specific adversarial patterns. We also apply gradient clipping to stabilize training when adversarial examples produce large loss values early in training.

We also need to adjust epoch count because the model must learn both clean classification and adversarial robustness. Standard training might converge in 10 epochs; adversarial training typically requires 20-30 epochs to achieve stable robustness. Early epochs show rapid improvement in clean accuracy while robustness lags. Later epochs refine the decision boundaries to improve robustness without sacrificing clean performance.

![Adversarial Training Progression](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/322/adversarial_training_training_progression.png)

Learning rate scheduling improves final accuracy. Cosine annealing reduces the learning rate smoothly across training, allowing aggressive early updates that transition to fine-tuning as training progresses. Step decay (reducing learning rate at fixed intervals) also works well. Both approaches help the model settle into robust minima rather than oscillating around them.

---

## Computational Cost

Adversarial training costs roughly `2-3x standard training per epoch`. Each batch requires a forward and backward pass to generate adversarial examples (for the gradient computation), followed by another forward and backward pass for the actual training update. We also double the effective batch size (clean plus adversarial), which increases memory usage and computation in the training passes.

Despite this overhead, `adversarial training remains practical for most applications`. Training a robust MNIST classifier takes 5-10 minutes on a GPU, 20-30 minutes on CPU. Larger models like ResNet on CIFAR-10 might take several hours rather than the one hour required for standard training. Since the cost scales linearly with model and dataset size, adversarial training remains feasible even for production-scale systems.

Acceleration techniques can reduce the overhead. Mixed-precision training (FP16 computations) cuts memory and time roughly in half. Free adversarial training reuses gradients from the training backward pass to compute adversarial perturbations, avoiding the extra forward-backward pair. These optimizations are valuable for large-scale deployments but unnecessary for the MNIST classifier in this lab.

---

## Limitations and Extensions

Adversarial training provides strong defense against the specific attack used during training but does not guarantee robustness against all possible attacks. A model trained against FGSM may remain vulnerable to more advanced attacks that optimize perturbations more carefully, such as the `Jacobian-based Saliency Map Attack (JSMA)`, which we explored in the [AI Evasion - Sparsity Attacks](https://academy.hackthebox.com/module/details/320) module. Training against stronger attacks improves generalization but increases computational cost.

Transfer attacks pose a particular challenge. An adversary might train their own model, generate adversarial examples against it, and apply those examples to the target model. If the adversary's model differs architecturally or trains on different data, their adversarial examples might exploit vulnerabilities the target never trained against. Ensemble adversarial training (generating attacks against multiple models) provides partial defense.

Certified defenses offer a different approach. Rather than training against specific attacks, certified methods provide mathematical guarantees that no attack within the epsilon ball can cause misclassification. Randomized smoothing and interval bound propagation provide certificates with varying tightness. These methods typically achieve lower robust accuracy than adversarial training but guarantee their results.

The next section walks through implementing adversarial training in the lab environment, from setting up the starter code through training a robust model that meets the evaluation criteria.

