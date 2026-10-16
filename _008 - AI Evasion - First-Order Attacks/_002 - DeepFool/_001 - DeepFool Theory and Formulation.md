
The previous section established DeepFool's geometric foundations through binary classification and iterative linearization. The algorithm projects inputs orthogonally onto linearized decision boundaries, accumulating perturbations until the true boundary is crossed. Real-world classifiers, however, operate on many classes simultaneously. How does this geometric framework extend when multiple decision boundaries surround each point?

## Multi-Class Formulation

The iterative linearization strategy works well for binary classification, where only one decision boundary separates the classes. Real-world classifiers typically handle many classes simultaneously. MNIST recognizes 10 digits. ImageNet classifies 1000 object categories. From any point in the input space, multiple decision boundaries exist, each separating the current predicted class from a different alternative. Which boundary should DeepFool target?

The answer: whichever is closest. Rather than committing to a predetermined target class, DeepFool dynamically identifies the nearest boundary at each iteration and takes the minimal step toward it. This adaptive strategy ensures the final perturbation approximates the minimal perturbation needed to cross any decision boundary, not just a specific one.

For an input x classified as class k̂(x)=arg⁡max⁡kfk(x), where fk(x) denotes the score for class k, DeepFool seeks the minimal perturbation r such that k̂(x+r)≠k̂(x). At each iteration, for every alternative class k≠k̂(xi), define the gradient difference wk=∇fk(xi)−∇fk̂(xi)(xi) and the score gap fk′=fk(xi)−fk̂(xi)(xi). The vector wk points in a direction that increases class k’s score while decreasing the current class’s score. For instance, if the current class score is 8 and class k’s score is 3, then fk′=3−8=−5, indicating a five-point deficit to overcome.

The closest boundary is identified by l=arg⁡min⁡k≠k̂(xi)|fk′|||wk||2, where the numerator |fk′| is the gap to close and the denominator ||wk||2 reflects how quickly that gap can be closed along direction wk. The minimal step is then ri=|fl′|||wl||22wl, the orthogonal projection onto the closest linearized boundary, with ||wl||22 normalizing by sensitivity. The perturbed input updates as xi+1=xi+ri before re-linearizing if the classification has not yet changed.

This formulation generalizes the binary case naturally. At each iteration, DeepFool evaluates all alternative classes, computes the distance to each boundary, and takes the minimal step toward the closest one. The geometric intuition remains identical: find the shortest path. The multi-class extension simply adds a selection step to identify which of many boundaries lies nearest.

## The Overshoot Parameter

DeepFool includes an `overshoot` parameter (typically 0.02) that slightly overshoots the decision boundary: the actual step taken is (1+overshoot)×ri. This serves two purposes:

The linearization at each point is only locally accurate. A small overshoot ensures we actually cross the non-linear boundary rather than merely touching it in the linear approximation. Without this, numerical precision issues could leave us infinitesimally close to but not across the boundary.

The overshoot also accelerates convergence by taking slightly larger steps, reducing the number of iterations needed. The trade-off is a marginally larger final perturbation, but the 2% typical value keeps this increase negligible while improving reliability.

## Comparison with Iterative FGSM

Both methods iterate. Both compute gradients multiple times. Yet their algorithmic approaches differ fundamentally in how they use gradient information.

`Iterative FGSM` takes uniform steps in the direction of the gradient sign. Each iteration applies the same step size across all pixels, then projects the result back onto the constraint set. The sign operation discards magnitude information, treating a gradient component of 0.001 identically to one of 100.

DeepFool preserves gradient magnitudes to compute geometrically minimal steps. Each iteration identifies the closest decision boundary among all classes and takes exactly the step needed to reach that specific boundary in the linearized approximation. Features with large gradient components naturally receive larger perturbations, while features with small gradients change minimally. This adaptive weighting emerges automatically from the orthogonal projection formula rather than being imposed through hyperparameter selection.

## Robustness Evaluation

DeepFool provides a quantitative measure of network robustness through the average minimal perturbation required to fool the classifier. This metric, often denoted as ρadv, is computed as:

ρadv=1|D|∑x∈D||r(x)||2||x||2

where D is a dataset and r(x) is the minimal perturbation found by DeepFool for input x.

This formula computes the average "relative perturbation size" across a dataset. For each image, the size of the perturbation ||r(x)||2 is divided by the size of the original image ||x||2 to obtain a percentage-like measure. If an image with norm 100 requires a perturbation with norm 2 to fool the classifier, that corresponds to a 2% relative perturbation. These percentages are then averaged across all images in the dataset D. A model with ρadv=0.02 requires, on average, perturbations that are 2% the size of the original input to be fooled, while a model with ρadv=0.10 requires 10% perturbations, making it five times more robust.

This robustness measure has several advantages over other metrics. It directly quantifies the distance to decision boundaries, provides a normalized measure that accounts for input magnitude, and can be computed efficiently for large datasets. Networks with larger ρadv values are generally more robust to adversarial perturbations.

## Extensions and Variations

The core DeepFool algorithm uses the L2 norm, but the methodology extends to other settings. The L∞ version finds the minimal perturbation in terms of maximum absolute change per pixel. For L∞ DeepFool, the distance computation changes:

l̂=arg⁡min⁡k≠k̂(xi)|fk′|∥wk∥1,ri=|fl̂′|∥wl̂∥1sign(wl̂).

The denominator switches from L2 norm to L1 norm, and the perturbation direction uses element-wise sign rather than normalized gradient. This distributes the perturbation uniformly across pixels, with magnitude |fl′|‖wl‖1 setting the amount needed to reach the boundary under the L∞ constraint.

Building on DeepFool's minimal perturbation principle, researchers developed `universal adversarial perturbations`: single perturbations that fool a network on most inputs from a dataset. The algorithm iteratively applies DeepFool to different training examples and accumulates the perturbations, with constraints on total magnitude. The existence of universal perturbations demonstrates that deep neural networks exhibit systematic vulnerabilities consistent across different inputs.

