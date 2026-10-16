
The previous section configured our environment and trained a target model. Now we develop the mathematical machinery needed to optimize ElasticNet's mixed-norm objective. The question is far from trivial.

FGSM had it easy. Closed-form solution. Sign operations. Done. DeepFool repeatedly solved linear approximations using closed-form projection formulas. ElasticNet’s mixed objective admits no such simple solution. The L1 term creates a fundamental obstacle.

Why is L1 optimization so difficult? The absolute value function |x| has a sharp corner at zero where the derivative is undefined. Standard gradient descent relies on smooth, continuous derivatives to determine update directions. At exactly the points where sparsity occurs (values becoming zero), the gradient doesn’t exist. Attempting naive gradient descent on ∥δ∥1 produces frustrating results: values hover near zero, getting smaller and smaller, but rarely reach zero exactly. You get pseudo-sparsity (many small values) instead of true sparsity (many exact zeros).

The `Fast Iterative Shrinkage-Thresholding Algorithm` (FISTA) decomposes the problem into smooth and non-smooth components, handling each with appropriate techniques. Proximal operators replace gradients for non-smooth terms, while Nesterov momentum accelerates convergence. Together, these tools enable efficient optimization of objectives that would defeat simpler methods.

## The Proximal Operator Framework

How do we generalize projection to handle non-smooth penalty functions? Proximal operators provide the answer. To understand this generalization, consider first the simpler case of constrained optimization that appeared in FGSM and DeepFool.

FGSM can be viewed as solving the linearized objective under an L∞ constraint: maximize ∇J(x)⋅δ subject to ∥δ∥∞≤ϵ. The solution is δ=ϵ⋅sign⁡(∇J(x)). DeepFool’s linear approximation similarly projects onto hyperplanes representing decision boundaries, using the geometric formula for distance from a point to a plane.

Proximal operators extend this projection idea to handle penalty functions rather than hard constraints. Instead of asking "what’s the closest point in set C?", we ask "what point minimizes the sum of distance-from-here plus some penalty function?" Formally, for function h at point z with step size λ:

proxλh(z)=arg⁡min⁡x{12∥x−z∥22+λh(x)}

What do these terms accomplish? The first term 12∥x−z∥22 measures squared distance from the input point z, encouraging solutions close to z. The second term λh(x) applies the penalty function, encouraging solutions with desirable properties according to h. The minimizer balances these competing objectives, finding a point that is both close to z and has small penalty under h.

Different penalty functions produce different proximal operators. When h is an indicator function (infinite outside a set, zero inside), we recover ordinary projection. When h is a differentiable function like ∥x∥22, we get a closed form involving gradient steps. When h is non-smooth like ∥x∥1, the proximal operator provides a well-defined operation that replaces the problematic gradient.

For the L1 norm, the proximal operator has a particularly elegant form: element-wise soft thresholding. This operation treats each coordinate independently, applying a simple threshold rule that zeros out small values and shrinks large ones. The threshold level λ controls how aggressively sparsity is enforced: larger λ creates sparser solutions by zeroing out more coordinates.

## Soft Thresholding: The L1 Proximal Operator

What is the proximal operator for h(x)=∥x∥1? The answer is elegant: soft thresholding. The operator 𝒮λ acts element-wise according to this rule:

𝒮λ(z)i={zi−λifzi>λ0if|zi|≤λzi+λifzi<−λ

To see the operator in action, set λ=0.1. For zi=0.12, soft thresholding returns 𝒮0.1(z)i=0.02; for zi=0.08, it returns 0; for zi=−0.25, it returns −0.15. In vector form, for z=[0.12,0.08,−0.25] we get 𝒮0.1(z)=[0.02,0.0,−0.15]. This simple check illustrates the three-region rule: exact zeros near the origin and shrinkage just outside that zone.

Three regions. Three actions. Values with magnitude below λ get zeroed out completely, creating exact sparsity. Positive values above λ are reduced by λ, shrinking toward zero from above. Negative values below −λ are increased by λ, shrinking toward zero from below. The name "soft thresholding" distinguishes this from hard thresholding, which would zero out values below λ but leave others unchanged (imagine cutting with scissors for hard thresholding versus gradually squeezing for soft).

Why does this operation minimize the proximal objective? Consider a single coordinate of the optimization problem. We want to find xi that minimizes 12(xi−zi)2+λ|xi|. The first term is a quadratic pulling xi toward zi. The second term is an absolute value penalty pulling xi toward zero. These competing forces balance at the soft threshold. To see this mathematically, split the problem into three cases based on zi.

For zi>λ, assume the minimum occurs at positive xi>0, making |xi|=xi. Taking the derivative and setting to zero: xi−zi+λ=0, giving xi=zi−λ. This confirms the first case of soft thresholding. For zi<−λ, assuming negative xi<0 makes |xi|=−xi. The derivative equation becomes xi−zi−λ=0, giving xi=zi+λ, confirming the third case. For |zi|≤λ, we can verify that xi=0 achieves lower objective value than any non-zero xi, confirming the middle case.

When zi is small (magnitude less than λ), the L1 penalty’s pull toward zero dominates the quadratic’s pull toward zi. The minimum occurs at exactly zero. When zi is large, the quadratic term dominates near zi, but the L1 penalty still exerts constant downward pressure, causing the minimum to occur at zi offset by λ toward zero. This analysis confirms that soft thresholding solves the one-dimensional proximal problem exactly.

The full proximal operator applies this element-wise operation across all coordinates. Because the L1 norm decomposes as a sum of absolute values, and the squared L2 norm similarly decomposes as a sum of squares, the multi-dimensional proximal problem separates into independent one-dimensional problems. Each coordinate can be thresholded independently without considering the others, making the operation computationally efficient and easily parallelizable.

Contrast this with the L2 proximal operator, which must consider all coordinates jointly. The proximal operator for h(x)=∥x∥2 produces proxλh(z)=max⁡(0,1−λ/∥z∥2)⋅z, a scaling operation that treats the entire vector as a unit. This operator shrinks vectors uniformly rather than creating sparsity by zeroing individual coordinates.

## From Proximal Operators to FISTA

Proximal gradient methods solve problems of the form min⁡x{f(x)+h(x)} where f is smooth but h is not. The strategy decomposes each iteration into two steps. First, take a standard gradient step on the smooth function f, moving from current point x(k) in the direction −∇f(x(k)) with step size η. Second, apply the proximal operator of h to handle the non-smooth term.

The basic proximal gradient update is:

x(k+1)=proxηh(x(k)−η∇f(x(k)))

This iteration provably converges to a minimizer under standard assumptions on f and h. The convergence rate is O(1/k) where k is the iteration count, meaning the error decreases inversely with iterations. While this beats naive subgradient methods, it remains relatively slow for problems requiring high precision.

FISTA accelerates this basic scheme using Nesterov momentum. Instead of computing gradients at the current point x(k), FISTA computes them at an extrapolated point y(k) that anticipates where the optimization is heading. This look-ahead allows the algorithm to better exploit consistent directions in the objective landscape, achieving faster convergence.

The FISTA update sequence is:

x(k+1)=proxηh(y(k)−η∇f(y(k)))

tk+1=1+1+4tk22

y(k+1)=x(k+1)+tk−1tk+1(x(k+1)−x(k))

The variable tk controls the momentum coefficient. Starting from t0=1, this sequence grows with iterations, providing progressively stronger acceleration. The specific formula for tk is carefully chosen to ensure the convergence guarantee: FISTA achieves O(1/k2) convergence rate, a quadratic improvement over standard proximal gradient descent.

In practice, the momentum update can be simplified. The papers introducing FISTA showed that tk≈k/2 for large k, leading to momentum coefficient (tk−1)/tk+1≈(k−2)/(k+1). For implementation simplicity, many practitioners use the approximation k/(k+3), which provides similar acceleration behavior while being slightly more conservative. This approximation maintains fast convergence while improving numerical stability.

## Applying FISTA to ElasticNet

Now we apply FISTA to ElasticNet's specific attack objective, decomposing it into components that optimization can handle efficiently.

ElasticNet pairs adversarial loss, which drives misclassification, with mixed-norm distance penalties that control perturbation size and sparsity. We use the Carlini & Wagner margin formulation, measuring how much the true class logit Zy(x′) exceeds the strongest competitor. The distance term combines squared L2 for smoothness with L1 for sparsity, weighted by β. A trade-off constant c balances these objectives.

For ElasticNet, decomposing into smooth and non-smooth parts follows naturally from this structure. Define the smooth part f as the adversarial loss plus the L2 distance:

f(x′)=c⋅max⁡(Zy(x′)−max⁡j≠yZj(x′)+κ,0)+∥x′−x∥22

Define the non-smooth part h as the L1 penalty:

h(x′)=β∥x′−x∥1

Gradients for f combine backpropagation through the neural network (for the adversarial loss) with a simple linear gradient (for the L2 term). Both components are differentiable almost everywhere, with the adversarial loss’s hinge being handled by standard subgradient conventions. The L2 gradient is particularly simple: ∇x′∥x′−x∥22=2(x′−x).

Soft thresholding is the proximal operator for h, applied to the perturbation. With δ=x′−x for the perturbation, we have:

proxηβ∥⋅∥1(z)=x+𝒮ηβ(z−x)

This formulation makes explicit that sparsity is enforced on the perturbation itself, not the adversarial image. We want sparse perturbations (many pixels unchanged), not sparse images (many pixels zero).

Each FISTA iteration performs these steps in sequence. Compute the gradient of smooth terms at the extrapolated point y(k). Take a gradient step with learning rate η. Apply soft thresholding with threshold ηβ to enforce L1 sparsity. Update the momentum point y(k+1) using the simplified formula k/(k+3). This sequence repeats for hundreds of iterations until convergence, with η controlling the gradient step size.

To see how the proximal step and momentum interact, consider a two-dimensional example. After a gradient step, suppose the candidate perturbation relative to the original is δ=[0.12,−0.07] and set β=0.1. Soft thresholding gives 𝒮0.1(δ)=[0.02,0.0]. The first coordinate survives with shrinkage; the second drops to zero, which enforces sparsity on small changes. With k=10, the momentum coefficient k/(k+3)=10/13≈0.769 extrapolates the next evaluation point along the update direction, using recent progress to look ahead and speed convergence when updates keep pointing the same way.

## Why FISTA for ElasticNet?

ElasticNet's iterative optimization differs fundamentally from FGSM's single-step approach and DeepFool's geometric linearization. The specific optimization mechanics make this difference clear.

FGSM requires no iterative optimization. One gradient evaluation yields the attack direction through a sign operation. Total iterations: one. DeepFool iterates through geometric projections, solving 5-20 closed-form linear problems without gradient descent or line search. FISTA performs true iterative optimization through gradient descent with proximal operators, requiring 100-1000 iterations where each step computes gradients via full backpropagation, applies soft thresholding for L1 sparsity, and updates momentum for acceleration.

Why this complexity? ElasticNet’s mixed L1/L2 objective admits no closed-form solution. The L1 term creates non-smooth corners where standard gradients fail. The misclassification constraint couples perturbations to the neural network’s decision boundary in ways that resist geometric simplification. FISTA handles this complexity by decomposing the problem into manageable pieces: gradient descent for smooth terms, proximal operators for non-smooth terms, and momentum for acceleration.

The following sections build toward the complete FISTA implementation. We begin by defining the loss functions that provide gradients: distance metrics quantify perturbation magnitude, adversarial loss drives misclassification, and the total loss combines these components. With loss functions established, we then implement the FISTA building blocks: momentum computation, soft thresholding, and the complete iteration step. Finally, we integrate these pieces with binary search to produce minimal adversarial perturbations.

