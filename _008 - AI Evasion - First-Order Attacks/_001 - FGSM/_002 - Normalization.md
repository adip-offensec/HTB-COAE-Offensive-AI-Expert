
# Normalization

Before proceeding with FGSM implementation, we need to understand what normalization does and why it matters for both training and adversarial attacks. Normalization transforms input data to have zero mean and unit variance, which fundamentally changes how we interpret perturbation budgets.

## The Operation

Normalization applies a simple linear transformation to each pixel. Given a dataset with mean μ and standard deviation σ, the normalized value xnorm is computed from the original pixel value x as:

xnorm=x−μσ

The subtraction x−μ centers the data around zero, removing the baseline brightness level. The division by σ scales the data to unit variance, ensuring similar ranges across different datasets. For MNIST, the library uses μ=0.1307 and σ=0.3081, which are computed from the entire training set of 60,000 images.

How are these statistics calculated? The mean μ=0.1307 represents the average pixel intensity across all training images in [0,1] range. Since MNIST digits are mostly black background (value 0) with white foreground (value 1), this low mean makes sense. The standard deviation σ=0.3081 measures how much pixel values vary around that mean. These specific values are standard for MNIST and should be used consistently across all experiments for reproducibility.

## Why Normalize?

Here’s the confusing part: normalized and unnormalized images look `identical` to us. A digit 7 in pixel values [0,1] looks exactly like the same digit after normalization to [−0.42,2.82]. Visually, nothing changed. So why bother?

The answer lies in the neural network's `gradient mathematics`, not human perception. This matters critically for adversarial attacks because FGSM relies entirely on computing gradients with respect to inputs. If those gradients behave poorly during training, they behave poorly during attacks too. Understanding how normalization affects gradient flow reveals why attacks work differently on normalized versus unnormalized models.

### Training Without Normalization

Consider training a simple neural network on unnormalized MNIST pixels in [0,1]. During backpropagation, the gradient magnitude depends on both the input values and the weight values. When inputs are all positive (pixel values between 0 and 1), gradients for weights in the first layer all point in similar directions because ∂loss∂w=x⋅δ where x>0 always.

This creates three problems. First, the optimizer takes tiny, inefficient steps because all gradients point in correlated directions rather than exploring the full parameter space. A network that could converge in 10 epochs might take 30 epochs instead. Second, different layers receive vastly different gradient magnitudes. Early layers might get gradients around 0.001 while later layers get gradients around 10.0, forcing you to use layer-specific learning rates or adaptive optimizers. Third, using a single learning rate becomes a balancing act. Set it too high and late layers explode. Set it too low and early layers barely move.

What does this look like in practice? Training on unnormalized MNIST with a simple CNN and learning rate 0.001 produces slow convergence. At epoch 1, the loss starts at 2.3 with accuracy around 10% (random guessing). By epoch 5, loss drops to 1.8 with 35% accuracy, showing slow progress. At epoch 10, loss reaches 1.2 with 65% accuracy, still climbing slowly. Only at epoch 20 does the loss hit 0.6 with 85% accuracy. The model eventually learns, but painfully slowly, and gradients fluctuate wildly during training.

### Training With Normalization

Now normalize those same images using (x−0.1307)/0.3081. A raw MNIST pixel with value 0.8 (bright white on the digit) becomes (0.8−0.1307)/0.3081≈2.17. A background pixel with value 0.0 (black) becomes (0.0−0.1307)/0.3081≈−0.42. The transformed values span roughly [−0.42,2.82] instead of [0,1], with the new distribution centered near zero.

What changed mathematically? The inputs now have zero mean and unit variance. This means gradients in early layers receive both positive and negative signals, breaking the correlation. Weight updates explore the parameter space more efficiently. All layers receive gradient magnitudes in similar ranges because the input statistics are standardized. The optimizer can use a single learning rate across all layers.

The difference in training speed is substantial. Training the same CNN on normalized MNIST with the same learning rate 0.001 shows rapid convergence. At epoch 1, loss drops immediately to 0.4 with 88% accuracy, a large jump compared to the unnormalized baseline. By epoch 5, loss reaches 0.08 with 97.5% accuracy, showing rapid convergence. At epoch 10, loss hits 0.04 with 98.5% accuracy, strong performance. The model reaches 98% accuracy in one epoch versus 20 epochs without normalization. Training time drops from minutes to seconds. Gradient magnitudes stay stable throughout training instead of spiking and crashing.

![Line charts comparing training accuracy: unnormalized inputs reach 85% by epoch 20 while normalized data hits 98% after a single epoch.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/FGSM_training_speed_simple.png)

### Training Speed and Attack Vulnerability

This rapid convergence has a surprising consequence for adversarial robustness. The faster training creates sharper, more confident decision boundaries. Sharp boundaries mean small input perturbations can push samples across them more easily. An underfit model (like the unnormalized one at epoch 5 with 35% accuracy) has fuzzy, uncertain boundaries. Attacking it with FGSM often fails because the gradients point nowhere useful. The model hasn't learned crisp features yet.

A well-trained normalized model has confident, sharp boundaries, making it simultaneously more accurate on clean data and more vulnerable to adversarial examples. The better the model, the easier it is to attack with gradient methods. This paradox drives modern adversarial robustness research: we want accurate models, but accuracy and vulnerability often go hand in hand.

### The Underlying Mechanism

Activation functions like ReLU and optimization algorithms like SGD are designed assuming inputs are roughly zero-centered with similar scales. When you feed ReLU an input of 0.7, it outputs 0.7. When you feed it 2.17, it still outputs 2.17, but the `derivatives` behave better because the distribution of activations across the layer has mean closer to zero. This prevents the `dead ReLU problem` where neurons get stuck outputting zero because all inputs are positive and push weights into negative territory.

For optimization, zero-centered inputs mean weight gradients don't all point in the same direction. Instead of all gradients being positive (causing slow, correlated updates), you get a mix of positive and negative gradients that explore the loss surface more efficiently. The standardized variance means the optimizer doesn't need to guess whether a gradient of 0.01 is large or small because it's calibrated to the input distribution.

![Scatter plots of weight gradients showing unnormalized updates confined near a line versus normalized gradients spreading across both axes.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/FGSM_gradient_exploration.png)

### Gradient Quality and Attack Success

Consider two identical models, one trained on unnormalized [0,1] inputs and one on normalized inputs. When computing input gradients for FGSM, the normalized model yields gradients that explore the full parameter space because training saw both positive and negative input-space values. The unnormalized model’s gradients remain correlated in direction because all training inputs were positive, limiting how effectively FGSM can probe decision boundaries.

This translates to real attack differences. On a normalized model, `epsilon=0.8` in normalized space (about 0.25 in pixel space) might achieve 95% attack success. The same pixel-space perturbation on an unnormalized model achieves only 60% success because the gradient directions are less informative. The attack isn't inherently weaker, the model's gradient landscape is just less structured.

This improvement compounds with network depth. A 2-layer network might train reasonably well without normalization. A 10-layer network will likely fail to converge at all without normalized inputs, because gradient magnitudes either explode (growing exponentially through layers) or vanish (shrinking to zero). Normalization keeps gradient flow stable even through dozens of layers, which also means adversarial gradients remain informative throughout the network.

## Range Transformation and Valid Bounds

When we normalize MNIST, the valid input range changes from [0,1] to approximately [−0.424,2.821]. These bounds come from applying the normalization formula to the original limits:

xmin=0.0−0.13070.3081≈−0.424

xmax=1.0−0.13070.3081≈2.821

These constants appear throughout the attack implementations as `MNIST_NORM_MIN` and `MNIST_NORM_MAX`. When we clamp adversarial images to these bounds, we ensure they remain valid MNIST inputs that can be converted back to [0,1] pixel space for visualization without clipping artifacts.

## Impact on Epsilon Budgets

Normalization fundamentally changes how we interpret ϵ budgets in FGSM attacks. When inputs are normalized, ϵ values are in normalized units unless explicitly stated. The relationship between normalized-space and pixel-space budgets is:

ϵpixel=σ⋅ϵnorm

For MNIST with σ=0.3081, an ϵnorm=0.8 in normalized space corresponds to ϵpixel=0.8×0.3081≈0.25 in [0,1] pixel space. This equals roughly 0.25×255≈64 intensity levels at 8-bit precision.

![Bar charts converting epsilon budgets between normalized and pixel spaces, noting that ε_norm 0.8 corresponds to about 0.25 in pixel magnitude.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/fgsm_epsilon_budget_spaces.png)

### Why This Matters for FGSM

The FGSM update rule xadv=x+ϵ⋅sign(∇xℒ) relies on computing ∇xℒ, the gradient of loss with respect to the input. When the model was trained on normalized inputs, those gradients are computed in normalized space. The ϵ budget must be specified in the same space to have consistent meaning. An ϵ=0.3 in [0,1] pixel space translates to roughly ϵ=0.97 in MNIST’s normalized space because ϵnorm=ϵpixel/σ=0.3/0.3081≈0.97.

When we specify `epsilon=0.8` for an attack on normalized inputs, we’re allowing each pixel to change by up to 0.8 standard deviations from its original value. This translates to different absolute changes depending on the pixel’s location in the distribution. A change of 0.8 in normalized space represents about 25% of the full [0,1] range, which is visually subtle but mathematically significant for the model’s decision boundary.

For adversarial attacks, this conversion matters when visualizing results or comparing across different normalization schemes. Attack code typically works in normalized space (where the model operates), but visualization converts back to [0,1] pixel space (where humans perceive images). An adversarial example with `epsilon=0.8` in normalized space looks subtly perturbed when denormalized for display, even though the numerical change is nearly 3 standard deviations.

