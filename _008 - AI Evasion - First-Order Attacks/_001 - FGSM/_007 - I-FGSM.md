

The `Iterative Fast Gradient Sign Method` (I-FGSM), also known as the Basic Iterative Method (BIM), was introduced by Kurakin et al. in Adversarial Examples in the Physical World (2016) "linked below" as an extension of Goodfellow et al.’s original FGSM method. Instead of one large, well-aimed step, the algorithm takes several small, well-aimed steps. Each step follows the input gradient’s sign, then projects back to the allowed L∞ budget around the original image. The result is a refined adversarial that more reliably crosses decision boundaries with the same overall budget.

_[Adversarial Examples in the Physical World](https://arxiv.org/abs/1607.02533)_

The motivation is simple. A single first-order step is efficient but it approximates a curved landscape with a flat plane. By repeating small steps and re-evaluating the gradient at the updated point, the algorithm better tracks the local geometry and often finds stronger perturbations at the same ϵ. The projection keeps the perturbation honest by enforcing the same per-pixel cap after every move, so improvements arise from better directions rather than larger budgets.

## Core Update and Projection

Starting from x(0)=x, the update for untargeted iterative FGSM is

x(t+1)=Πℬ∞(x,ϵ)(x(t)+αsign⁡(∇x(t)ℒ(θ,x(t),y))),

where α is the step size, typically α=ϵT for T iterations, and Πℬ∞(x,ϵ) projects to the L∞ ball of radius ϵ around x and then clips to the valid input domain. The projection is not an extra regularizer; it is the mathematical way to say "stay within the same per-pixel budget after each step." In coordinates, the projection is just per-pixel clipping:

Πℬ∞(x,ϵ)(x′)=x+clip⁡(x′−x,−ϵ,ϵ),x′←clip⁡(x′,xmin⁡,xmax⁡).

For targeted attacks, replace y by the target label yt and reverse the step direction so that the update increases the target’s score rather than the true label’s score. This is the same idea as targeted FGSM, applied repeatedly with projection so the budget remains ϵ throughout.

## Why Iteration Helps

The first step from x follows the gradient sign computed at x, which is the exact optimizer of the linearized inner maximization for the L∞ constraint. After stepping, the loss surface is no longer well-approximated by the original tangent plane. Recomputing the gradient at x(t) and taking another projected step adjusts to the new local geometry. Over several steps, this process tends to push examples closer to the true decision boundary while honoring the same budget ϵ.

For quick illustration, consider a case where FGSM with ϵ=0.8 moves the loss from 0.3 to 1.2 (a change of 0.9). With I-FGSM using 10 iterations and α=0.08, the loss might evolve as: 0.3 → 0.45 → 0.62 → 0.81 → 0.98 → 1.15 → 1.31 → 1.44 → 1.55 → 1.63 → 1.68, achieving a final loss of 1.68 (a change of 1.38). The iterative refinement yields 53% more loss increase with the same L∞ budget.

## Prerequisites

We'll build directly on the FGSM implementation from the previous sections. Ensure you have all the code from the FGSM Setup and Core Implementation sections, which provide the trained model, data loaders, and baseline attack functions.

For reference, you should have:

- A trained `SimpleCNN` model on MNIST with ~98% clean accuracy
- The `test_loader` and `device` configuration from the setup
- The `_input_gradient` and `fgsm_attack` functions from the FGSM implementation

