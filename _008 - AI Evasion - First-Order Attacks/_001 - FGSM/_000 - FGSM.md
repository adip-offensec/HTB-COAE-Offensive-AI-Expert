
# FGSM

The `Fast Gradient Sign Method (FGSM)` arrived as a bit of a wake-up call for the deep learning field.

In 2014, Goodfellow et al. introduced FGSM in "[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)," showing that a classifier that looks reliable on clean data can be flipped by tweaks so small that humans hardly notice. Start with an input the model labels correctly. Add a calculated nudge that pushes the network where it is vulnerable. The prediction changes. That simple recipe seeded modern adversarial machine learning.

The method turns the model's own learning signal against it. Instead of changing parameters to reduce error, the attack changes the input to increase it. The gradient computation reveals how the loss reacts to each pixel, keeping only the direction of that reaction to take a tiny, coordinated step. The mathematics below formalizes that idea, but the approach is straightforward: use gradients to find a direction that the network dislikes and move there, just a little.

This matters because it exposes fragility that accuracy alone hides. High-performing models in high-dimensional spaces can be sensitive to small, structured changes that human vision ignores. In settings that demand reliability, that sensitivity is a security concern and a design constraint.

We'll first see the core formula and its intuition, then understand why the attack succeeds through local linearity and high dimensionality, and finally explore the mathematical foundations that explain its optimality as a solution to a constrained optimization problem.

## FGSM Core Idea

![Diagram of FGSM linking the gradient-sign equation with the original digit, the perturbation map, the adversarial digit, and a histogram illustrating sign scaling.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/fgsm_core_idea.png)

In plain terms, the gradient is a map of sensitivities: it indicates which way to nudge each pixel to make the loss increase the fastest. The sign operation extracts only the direction of that nudge for each pixel and discards how large the change would be, so the algorithm changes every pixel by the same small amount ϵ but in the direction that increases the loss. The L∞ budget limits the maximum absolute change to any single pixel, not the average change across all pixels. (See the Introduction’s Understanding Norms section for the explanations of L∞, L2, L1, and their geometric interpretations if you need a reminder.)

Formally, `FGSM` produces adversarial examples by taking a single step in input space that maximally increases model loss while keeping every pixel change bounded by an L∞ budget. With input x, label y, parameters θ, and loss ℒ(θ,x,y), the adversarial input is

xadv=x+ϵsign⁡(∇xℒ(θ,x,y)).

This expression leverages local linearity around x. A first-order Taylor view suggests that the steepest ascent in loss arises along the gradient direction. Taking only the sign of the gradient equalizes step size across pixels, so the per-pixel change is exactly ϵ in magnitude, which fits the L∞ constraint. The parameter ϵ is the perturbation budget that trades off imperceptibility and attack strength.

![Three panels illustrate increasing L∞ budgets where every pixel change stays bounded, producing progressively noisier squares inside circular masks.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/Linf_max_change.png)

## Why FGSM Works

FGSM works for two simple reasons: `local linearity` and `high dimensionality`. Near a clean image, deep nets behave almost linearly (small input changes produce proportional score changes), so one `gradient-based` step already pushes the loss in the steepest direction. And because images have many pixels, tiny, correctly aligned tweaks at each one add up to a large change in the model’s score. The L∞ constraint caps every pixel’s change at ϵ, giving a practical per-pixel visibility limit while still letting those many small tweaks accumulate.

To make this precise, consider a simple linear model where the accumulation effect becomes mathematically explicit.

For a linear score function f(x)=w⊤x, the logit change caused by any perturbation with ∥δ∥∞≤ϵ satisfies the bound

|f(x+δ)−f(x)|=|w⊤δ|≤ϵ∥w∥1.

In high dimensions, ∥w∥1 can be large even when individual components are modest, which illustrates how small per-pixel changes aligned with the gradient can accumulate into a substantial output shift. For example, if an image has 784 pixels (28×28 MNIST) and each weight component has magnitude 0.01, then ∥w∥1=784×0.01=7.84. With ϵ=0.1, the maximum logit change is 0.1×7.84=0.784, which can easily flip a decision.

This inequality matters because it turns the vague idea of high-dimensional sensitivity into a concrete statement. Even though each pixel changes by at most ϵ, the aggregate effect grows with how many dimensions contribute and how strongly they align with the gradient. That is why small, structured perturbations can move a classifier’s decision.

## Constrained Linearization

Why does FGSM work so well? One view comes from optimization theory. The algorithm solves a specific problem exactly: finding the worst perturbation within our budget. The attack essentially asks "which direction hurts the model most?" and the sign-based step provides a precise answer. To see this, we approximate the curved loss surface with a flat plane that touches it at our starting point, turning a hard problem into one with a known solution.

The FGSM update arises as the exact optimizer of a first-order approximation to the inner maximization problem. Consider the adversarial objective that seeks a perturbation within an L∞ ball that maximizes the loss, written as

max⁡δ∈ℝnℒ(θ,x+δ,y)s.t.∥δ∥∞≤ϵ.

Using a first-order Taylor expansion around x yields the approximation

ℒ(θ,x+δ,y)≈ℒ(θ,x,y)+∇xℒ(θ,x,y)⊤δ.

The constant term does not affect the maximizer, so the approximate problem reduces to

max⁡∥δ∥∞≤ϵg⊤δwithg=∇xℒ(θ,x,y).

To find this maximum, we need a fundamental result from analysis. `Hölder’s inequality` states that for vectors u and v and conjugate exponents p and q satisfying 1p+1q=1, the dot product is bounded by |u⊤v|≤∥u∥p∥v∥q. The specific case relevant here pairs L∞ with L1 as dual norms (since 1∞+11=0+1=1), giving the bound |g⊤δ|≤∥δ∥∞∥g∥1. When ∥δ∥∞=ϵ, this becomes g⊤δ≤ϵ∥g∥1, and equality holds exactly when δ is aligned component-wise with the sign of g.

By Hölder’s inequality and L∞–L1 duality, the optimal value equals ϵ∥g∥1 and is attained at the vertex that aligns each component with the sign of the gradient. Specifically,

δ*=ϵsign⁡(g),max⁡∥δ∥∞≤ϵg⊤δ=ϵ∥g∥1.

Substituting g=∇xℒ(θ,x,y) gives the FGSM update xadv=x+δ*, which exactly solves the linearized constrained problem. The targeted variant follows by minimizing the loss towards a chosen target yt, equivalently maximizing the negative loss, which flips the sign of the step.

This linearization replaces the curved loss surface near x with a flat plane that just touches it at x. Maximizing a flat plane over a box-shaped region (the L∞ ball) pushes each coordinate of the perturbation to its boundary with the same sign as the gradient at that coordinate. The duality language is a compact way of stating this alignment result.

## Targeted and Untargeted Formulations

The original paper presents the untargeted objective that increases the loss for the true label:

xadv=x+ϵsign⁡(∇xℒ(θ,x,y)).

The targeted variant, which became standard practice after the original work, decreases the loss for a desired target class yt:

xadvtarget=x−ϵsign⁡(∇xℒ(θ,x,yt)).

This distinction only affects the step direction and which label is used in the gradient computation. The L∞ budget and clipping remain unchanged.

## Dual Norm Perspective and Alternative Budgets

One way to remember these norms is to view L2 as overall energy across pixels, L1 as the total absolute change summed across pixels, and L∞ as the single largest change over all pixels. Duality means that if you constrain one kind of size for the perturbation, the worst-case dot product is controlled by the complementary kind of size for the gradient.

The FGSM sign step is specific to the L∞ constraint. For an L2-bounded perturbation of radius ϵ, the linearized maximizer takes the direction of the gradient normalized in L2, written as

δ2*=ϵg∥g∥2,xadv=x+δ2*.

For an L1-bounded perturbation, the linearized maximizer concentrates the budget on coordinates with largest absolute gradient components. In practice, a sparse update that assigns ±ϵ along the top coordinates by magnitude approximates this solution. These alternatives follow from the duality between Lp and Lq norms with 1p+1q=1.

## Backpropagation to Inputs

The reason for computing this input gradient in an attack is straightforward. During standard training, gradients flow with respect to the parameters to tell the optimizer how to change the model. In an adversarial attack, the parameters are frozen and the object being changed is the input itself. The attack therefore asks a different question: if the pixels were nudged a tiny amount, in which direction would the loss increase fastest. The answer to that question is exactly the vector ∇xℒ(θ,x,y).

This input gradient is the quantity that appears in the linearized inner maximization. The first-order objective becomes g⊤δ with g=∇xℒ(θ,x,y), so the algorithm must compute g to know how to choose δ. Backpropagation to inputs provides g efficiently in a single backward pass through the network with x treated as a variable and θ kept fixed. In practical terms, the implementation sets `requires_grad=True` on `x`, runs a forward pass to compute the loss, calls `loss.backward()`, and reads `x.grad` as the sensitivity map.

Let zi(x) denote the pre-softmax logit for class i and let pi(x)=softmax⁡(z(x))i. With cross-entropy loss ℒ(θ,x,y)=−log⁡py(x), the gradient with respect to input can be expressed via the chain rule as

∇xℒ(θ,x,y)=∑i(pi(x)−𝟙[i=y])∇xzi(x).

For a piecewise-linear network composed of affine layers and ReLU activations, the mapping from inputs to logits is affine within each activation region. Consequently, the Jacobians ∇xzi(x) are constant within a region, which explains why a first-order step aligned with the gradient can be so effective locally.

The coefficients pi(x)−𝟙[i=y] weight how each class’s logit Jacobian contributes. Note that for the true class y, the coefficient is py−1=−(1−py): as confidence grows (py→1), its magnitude shrinks, so this term does not dominate. The remaining terms (for i≠y) sum to 1−py and push to increase competing logits. Altogether ∇xℒ=(∑ipi∇xzi)−∇xzy points in the direction that most increases the loss by both decreasing the true-class logit and increasing others, within the local linear region.

For example, if py(x)=0.95 for the true class, the coefficient for i=y is 0.95−1=−0.05, which is small in magnitude. The non-true coefficients collectively sum to 1−py=0.05 and distribute across i≠y. The resulting gradient gently lowers zy and gently raises competing logits. By contrast, if py(x)=0.20, then py−1=−0.80 and the non-true coefficients sum to 0.80, yielding a much larger gradient that more forcefully reduces the true-class logit while boosting others.

