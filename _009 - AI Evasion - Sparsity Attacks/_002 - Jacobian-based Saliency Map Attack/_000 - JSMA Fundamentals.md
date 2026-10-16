
# JSMA Fundamentals

The `Jacobian-based Saliency Map Attack` (`JSMA`) identifies which specific pixels most strongly influence a model's decision, then modifies only those critical features. Papernot, McDaniel, Jha, Fredrikson, Celik, and Swami demonstrated in their 2016 paper "[The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528)" that targeted selection of high-influence pixels can fool models while changing remarkably few features - often just 20-40 pixels out of 784 in MNIST.

JSMA minimizes the L0 norm (the count of modified features) rather than perturbation magnitude. Individual pixels can jump dramatically - from black (0.0) to white (1.0) - if the saliency map identifies them as strategically valuable. This creates visually apparent dots or strokes rather than imperceptible noise, trading stealth for extreme sparsity.

## Understanding the Trade-offs: Sparsity Versus Magnitude

JSMA embodies a different philosophy from other adversarial attacks, optimizing for minimal feature modification count rather than minimal perturbation magnitude. This distinction creates a unique set of trade-offs that make JSMA ideal for certain threat models while less suitable for others.

The contrast becomes clear when we examine an example. FGSM constrains the L∞ norm, modifying all 784 pixels but each by at most ϵ=0.03, creating nearly invisible uniform noise across the entire image. DeepFool minimizes the L2 norm to find the smallest overall perturbation magnitude needed to cross a decision boundary, typically modifying most pixels with varying magnitudes. ElasticNet balances sparsity and magnitude, modifying 100-200 pixels with controlled changes. JSMA, by focusing on the L0 norm, allows individual pixels to change significantly (even saturating from black to white) as long as the total number of modified pixels remains small. Effective JSMA configurations can modify as few as 20-40 pixels out of 784 total, with each selected pixel potentially changing by the full step size per iteration or being selected multiple times before saturating.

What does sparsity mean for detectability? JSMA perturbations become visually apparent when examined closely, appearing as scattered dots or strokes rather than uniform noise. A defense system looking for widespread coordinated changes might miss these isolated modifications. The sparse nature also aids interpretation, revealing which specific features the model relies on most heavily for its decisions.

Computational cost grows linearly with the number of output classes. Why? We need one backward pass per class to build the complete Jacobian matrix. MNIST with 10 classes requires 10 backward passes per iteration, manageable on modern GPUs. ImageNet with 1000 classes demands 1000 backward passes per iteration, making the attack prohibitively expensive for real-time scenarios.

## Gradient Target And Model Choice

Computing gradients with respect to `logits` rather than `probabilities` preserves the target sensitivity `α` and competitor sum `β` as independent signals. This independence lets saliency sign constraints genuinely screen candidates based on their ability to boost the target while suppressing competitors. Post-softmax probabilities create a problem: because probabilities sum to one, differentiation forces `β = -α` by mathematical construction. This constraint collapses saliency into a squared target slope, eliminating the competitor suppression term entirely. We lose JSMA's intended behavior from the original paper.

Our demonstrations use a LeNet-like MNIST model because sparse, targeted attacks prove most reliable on this architecture. The shallow network structure with limited capacity makes individual pixels highly influential on the output decision. This mirrors the original paper's experimental setup and provides reproducible baselines. Deeper modern CNNs like ResNet distribute decision-making across many layers through skip connections and batch normalization, reducing individual pixel influence. Success rates drop substantially on these architectures, which explains why modern sparse attacks like ElasticNet and SparseFool have largely superseded JSMA for complex models.

## Intuition Through Examples: How Saliency Guides Selection

To build intuition for how JSMA selects features, let's walk through a simplified scenario. Imagine a toy model with just two input features and three output classes, where we want to force misclassification to class 2 (our target).

![Initial class scores: class 0 highest, class 1 mid, target class 2 lower](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jacobian_step1_initial.png)

The visualization above shows our starting point: Class 0 is currently winning with the highest score (0.6), followed by Class 1 (0.5), while our target Class 2 lags behind at 0.3. This is the challenge JSMA must overcome. How can we modify features to flip this ranking?

Suppose the Jacobian reveals the following sensitivities. For feature 1: ∂F2∂x1=0.6 (boosting the target by 0.6 units) and ∑j≠2∂Fj∂x1=−0.4 (decreasing other classes by 0.4 combined). This is ideal because feature 1 helps our target and hurts competitors, yielding a saliency score of 0.6×0.4=0.24.

![After increasing feature 1, target class 2 overtakes others while competitors drop](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jacobian_step2_feature1.png)

Look at the effect: the target (aquamarine) increases to 0.9, while both competitors (red) drop. The arrows visualize these movements. This is what we want because the target now wins.

For feature 2: ∂F2∂x2=0.3 and ∑j≠2∂Fj∂x2=0.5. While this helps the target somewhat, it helps competitors even more. This violates our sign condition, yielding zero saliency.

![Increasing feature 2 boosts competitors more than target; class 0 remains on top](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jacobian_step3_feature2.png)

The problem becomes clear: while the target increases slightly (yellow), the other classes increase even more (aquamarine with larger arrows). Class 0 increases to 0.9, maintaining its lead. Feature 2 strengthens the original prediction, which is why JSMA rejects it.

This selection logic extends naturally to real images. Consider attacking a digit classifier on an image of a "3". We can identify which pixels along the top horizontal stroke strongly influence the "3" score while barely affecting other digits. Similarly, we know certain pixels in the middle curve boost "8" while suppressing "3".

![MNIST example: strategic pixels transform a "3" into an "8"](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jacobian_step4_mnist.png)

The MNIST example shows this in action. Starting with "3" (left), JSMA identifies strategic pixels (center, hot colors) that distinguish "3" from "8". Specifically, we need to fill in the left side and strengthen the middle stroke. By modifying just these pixels, the digit can be transformed to "8" with carefully targeted changes.

## The Attack Algorithm: Iterative Refinement

Unlike single-step attacks like FGSM, JSMA operates iteratively with a complete decision cycle at each step. We compute the Jacobian to understand current sensitivities, construct the saliency map to identify important features, select and modify the best feature, then update the search space. Repetition continues until we achieve misclassification or exhaust the L0 budget.

![Flow chart of JSMA cycle: compute Jacobian, build saliency map, select feature, modify, update search space](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/320/jacobian_iterative_cycle.png)

Two parameters govern the attack’s behavior: step size θ determines modification magnitude per iteration, while feature budget γ limits total sparsity as a fraction of features. Larger θ values produce faster attacks with more visible perturbations, while smaller values require more iterations but create subtler modifications. The feature budget γ implements hard sparsity constraints, calculating the maximum number of modifiable pixels as γ×num_features. Subsequent sections present detailed mechanics and empirical evidence showing how these parameters affect success rates, iteration counts, and visual quality.

## Common Pitfalls and Implementation Wisdom

Several implementation details make the difference between effective JSMA and failure.

Gradient sign interpretation causes the most common conceptual confusion. For each feature, we evaluate both increasing and decreasing its value. A negative gradient for the target class doesn't discard the feature; instead, it indicates that decreasing that feature helps achieve our goal. We evaluate both directions and choose the one with higher saliency. This is why JSMA can "erase" features (set white pixels to black) as effectively as it can "add" features (set black pixels to white).

Search space mechanics require careful management. We must remove saturated or out-of-range features promptly to avoid wasted iterations. In practice, this means checking after each modification whether the pixel has reached `clip_min` (typically 0.0) or `clip_max` (typically 1.0) and masking it out.

The monotonic mask prevents reselecting saturated pixels. Subsequent sections demonstrate these mechanics through proper implementation.

Shape management requires careful attention because gradients and modifications operate in different coordinate systems. Computing the Jacobian flattens the input into a 1D array for efficient gradient extraction (a 1×1×28×28 MNIST image becomes a vector of length 784). Selecting pixel 352 in this flat representation corresponds to row 12, column 16 in the original 2D grid. We must maintain consistent flattening order throughout: channel first, then height, then width. PyTorch’s `view_as()` or `reshape()` methods restore the original structure after modifications, but only if we flatten and unflatten using the same convention.

Mixing flatten and reshape orders creates catastrophic bugs. Flatten in C-H-W order but reshape assuming H-W-C order? Pixel 352's modification ends up at the wrong spatial location. Worse, the attack appears to run correctly (no errors, reasonable saliency values) while modifying semantically meaningless pixels, producing zero success rate.

Memory consumption scales as classes multiplied by features, creating practical limits on attack feasibility. MNIST requires storing (10×784)=7,840 floating-point values for the Jacobian, roughly 31 KB at single precision. Modern GPUs handle this trivially, leaving memory for model parameters and batch processing.

ImageNet reveals the scaling challenge. With 224×224 RGB images and 1000 classes, we need (1000×224×224×3)=150 million values, consuming 600 MB just for the Jacobian. A typical ResNet-50 already occupies 2-3 GB for parameters and activations, leaving limited headroom. Chunking strategies become necessary: compute gradients for 100 classes at a time, accumulate results on CPU, repeat. This trades memory for computation time but enables attacks on hardware-constrained systems.

Debugging benefits from visualization at each iteration. We can plot the saliency map as a heatmap to verify that high-saliency regions align with our intuition about important features. Tracking the target class confidence over iterations ensures it generally increases.

If the attack stagnates, examine whether the search space is shrinking too quickly due to aggressive boundary saturation. This suggests a smaller θ value might help. The visualization section demonstrates these debugging techniques with actual examples.

