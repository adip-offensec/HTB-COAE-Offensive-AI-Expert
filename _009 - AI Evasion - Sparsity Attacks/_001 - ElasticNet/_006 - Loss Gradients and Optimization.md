
To optimize adversarial images, we need ∇x′ℒtotal with respect to input pixels, not model weights. Freezing the network and enabling gradients only on adversarial images makes `loss.backward()` compute exactly what we need. These input gradients reveal how each pixel affects both misclassification (through the adversarial term) and distortion (through the L2 term), guiding FISTA’s iterative refinement.

## Adversarial Loss Gradients

How do adversarial loss gradients differ from L2 gradients? The adversarial component flows through the entire neural network, composing derivatives backward from output to input via the chain rule. Each layer contributes its local derivative, building complexity as we move toward the input. ReLU activations create piecewise-linear regions where gradients remain constant within each region but jump discontinuously at boundaries. Which neurons activate depends on the current input, making the gradient landscape non-convex and highly structured.

FISTA treats these complex gradients as a black box. Provide an input, receive a loss value and gradient; no need to understand the network's internal structure. We follow gradients downhill toward loss minima while the proximal operator enforces sparsity. This black-box approach makes ElasticNet general, applicable to any differentiable model regardless of architecture.

What happens to gradients during a typical FISTA iteration? Starting with the forward pass, we compute logits that determine adversarial loss through the margin formula. For an example with true class 3, if the current adversarial image produces logits `[0.2, 0.5, 1.8, 2.1, 0.9, ...]`, then `real = 2.1` (logit for class 3) and `other = 1.8` (maximum competitor, class 2). The margin is 2.1−1.8=0.3, giving adversarial loss of 0.3 (assuming κ=0).

Moving backward, we compute how changing each pixel affects this margin. If increasing a particular pixel pushes class 2's logit higher while decreasing class 3's logit, that pixel receives a negative gradient (increase it to reduce the margin, which reduces the loss). Opposite effects produce positive gradients. Magnitude indicates sensitivity: large gradients mean small pixel changes produce big margin changes.

Meanwhile, the L2 gradient provides a simple restoring force pulling pixels back toward their original values. If the current adversarial image has perturbation δi=0.15 at pixel i, the L2 gradient component is 2×0.15=0.30 pulling that pixel back toward the original. This gradient grows linearly with perturbation magnitude, automatically scaling resistance based on how far we’ve strayed.

## Loss Landscape and Gradient Evolution

How does ElasticNet’s loss landscape differ from pure L2 or L∞ attacks? The adversarial loss component creates steep cliffs near decision boundaries where small perturbations cause large loss changes. Meanwhile, the L2 component provides a smooth restoring force with gradients proportional to perturbation magnitude. These competing forces create a loss surface with mixed curvature: steep in directions toward misclassification, shallow in directions perpendicular to the decision boundary.

Gradient magnitudes evolve predictably as optimization progresses. Early iterations (1-100) show large adversarial gradients (5-20 per pixel) dominating small L2 gradients (0.1-0.5 per pixel), driving aggressive exploration toward misclassification. The attack starts bold. Middle iterations (100-500) bring balance. Adversarial loss begins saturating, allowing L2 to pull back excessive perturbations, with both components contributing 1-5 per pixel. Late iterations (500-1000) enter refinement mode where adversarial loss has saturated completely, leaving only L2 minimization (0.1-1.0 per pixel) to reduce distortion while preserving misclassification.

The trade-off constant c amplifies adversarial gradients relative to L2 gradients. With c=0.001, adversarial gradients contribute minimally (0.001-0.05 per pixel), allowing L2 minimization to dominate; the attack might fail to achieve misclassification. With c=10, adversarial gradients dominate (50-500 per pixel), overwhelming the L2 restoring force. The attack succeeds easily but produces unnecessarily large perturbations. Binary search finds the sweet spot where adversarial pressure barely suffices for misclassification, typically c=0.01−1.0 for MNIST.

Why does c matter locally? Consider one pixel. If δ=0.15, the L2 gradient contributes 2×0.15=0.30 toward the original. Suppose the adversarial gradient at that pixel is −2.5 (reducing the margin). With c=0.1, the scaled adversarial term contributes −0.25, so the combined effect is 0.30−0.25=0.05 (a small net pull toward the original while still reducing the margin overall). With c=1.0, the same terms sum to 0.30−2.5=−2.20, flipping direction and pushing strongly toward misclassification. This local balance mirrors the global trade-off tuned by binary search.

Loss component interaction affects convergence behavior. When the adversarial loss saturates (saturates at 0), it contributes zero gradient, and optimization reduces to pure L2 minimization with soft thresholding. This automatic transition from "achieve misclassification" to "minimize distortion" occurs naturally through the hinge function, requiring no explicit mode switching. Before saturation, both components pull simultaneously, creating curved trajectories through perturbation space rather than straight lines.

Numerical issues occasionally arise when gradients become too large or too small. Extremely large gradients (>100 per pixel) during early iterations can cause instability where gradient steps overshoot, producing adversarial images outside valid bounds. The clipping operations in shrinkage thresholding prevent this from corrupting the perturbation, but it wastes iterations oscillating. Extremely small gradients (<0.001 per pixel) during late iterations signal convergence, but premature occurrence (iterations <100) indicates stagnation where the optimizer failed to escape a poor local minimum. Monitoring gradient magnitudes provides diagnostic information about optimization health.

## Gradient Flow Through Network Layers

Why do certain pixels receive stronger gradient signals than others? Understanding backward gradient flow through CNN architectures reveals these patterns. Consider a typical network with convolutional layers, ReLU activations, max pooling, and fully connected layers.

Starting at the output layer (before softmax), logits emerge through a linear transformation: Z=Wout⋅h+bout, where h is the hidden representation. The adversarial loss gradient ∂f∂Z flows backward through this layer using the transpose operation: ∂f∂h=WoutT⋅∂f∂Z. Hidden units with large outgoing weights to relevant classes receive stronger gradients, amplifying their influence on the final perturbation pattern.

Moving backward through ReLU activations, we encounter piecewise-linear behavior. When a unit is active (input > 0), gradients pass through unchanged. When inactive (input ≤ 0), gradients get blocked entirely. This creates discrete regions in input space with different gradient patterns. Moving an input across a ReLU boundary changes which units activate, potentially causing dramatic gradient shifts that make the loss landscape non-convex.

To distribute gradients spatially, convolutional layers spread each output gradient across the filter's receptive field. A gradient at one spatial location in a feature map distributes across all input pixels within that filter's window. For a 3×3 convolution, each output gradient affects a 3×3 patch of input pixels. Deep networks with large receptive fields (through multiple conv layers or pooling) can propagate gradients from output changes to input pixels far from the semantic feature that actually matters for classification.

Pooling introduces sparsity in backward connections. Only the maximum value within each pool window receives gradients; other values get zero gradient. This sparsity means many input pixels receive no direct gradient signal from certain output changes, potentially making the loss landscape less smooth and optimization more challenging. Pixels that never win the max operation contribute nothing to gradients despite occupying space in the input.

## Practical Implications for FISTA

These gradient properties have direct implications for FISTA optimization. The learning rate must be scaled to gradient magnitudes, as larger gradients require smaller step sizes to prevent overshooting.

The non-convex loss landscape means FISTA finds local minima, not global optima. Different initializations (starting from different random perturbations) might converge to different solutions with varying attack effectiveness. In practice, ElasticNet initializes from the original images (zero perturbation), providing a consistent starting point. The binary search then explores different constant values, effectively providing multiple restarts with different adversarial pressures.

Gradient sparsity from ReLU and max pooling means some pixels receive weak or zero gradients. Soft thresholding compounds this: pixels with weak gradients produce small perturbations that get zeroed during thresholding. This automatic filtering helps sparsity but might miss pixels that could contribute meaningfully if given stronger signals. The iterative nature of FISTA allows gradients to build up over many iterations, potentially activating pixels that initially received weak signals.

## Convergence Properties and Stopping Criteria

FISTA’s O(1/k2) convergence guarantee applies under standard assumptions: f must have Lipschitz continuous gradients, h must be convex, and the step size η must satisfy η≤1/L where L is the Lipschitz constant of ∇f. For neural networks, the Lipschitz constant depends on the model architecture, with deeper networks and larger weights producing larger constants.

In practice, these theoretical conditions are often violated. Neural network losses are non-convex, meaning the convergence guarantee technically doesn't apply. However, FISTA often works well empirically even on non-convex problems, finding good local minima efficiently. The algorithm may not find the global optimum, but the local solutions it discovers typically produce effective adversarial examples.

Stopping criteria determine when FISTA iterations terminate. The simplest approach runs a fixed number of iterations (e.g., 1000), chosen large enough to ensure reasonable convergence on most examples. This approach wastes computation on easy examples that converge quickly while potentially under-optimizing difficult ones that need more iterations.

More sophisticated stopping criteria check for convergence dynamically. One approach monitors the change in objective value between iterations, stopping when |f(k+1)+h(k+1)−f(k)−h(k)|<ϵ for some tolerance ϵ. Another checks whether the attack has succeeded (caused misclassification) and the adversarial loss has saturated (reached its minimum value of −κ), indicating that further iterations would only reduce distortion without affecting attack success.

The ElasticNet implementation uses a hybrid approach. FISTA runs for a maximum number of iterations per binary search step, with periodic checks for attack success. If all examples in a batch successfully fool the model early, that binary search iteration can terminate without completing all FISTA iterations. This adaptation prevents wasted computation while ensuring sufficient optimization for difficult examples.

The next section combines all these components into the complete FISTA iteration function. We'll see how gradient computation, soft thresholding, and momentum updates compose into a single optimization step. Then we'll add the success checking and binary search mechanisms that guide the attack toward minimal perturbations.

