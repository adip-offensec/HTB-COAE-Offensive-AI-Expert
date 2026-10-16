
# The Opacus Library

We use the `Opacus` library to implement DP-SGD. Opacus wraps your PyTorch optimizer and data loader with a `PrivacyEngine` that modifies gradient computation during each training step. Instead of computing the batch-average gradient, Opacus computes `per-sample gradients` and clips each to the configured `max_grad_norm`. The clipped gradients are then noised with Gaussian noise scaled to the desired privacy budget and tracked through a `privacy accountant` that monitors cumulative privacy expenditure.

To configure privacy, we call `make_private_with_epsilon()`, which automatically calculates the required `noise multiplier` for a target privacy budget. During training, `get_epsilon()` reports cumulative privacy expenditure at any point.

## Per-Sample Gradients and Architecture Constraints

Standard PyTorch backpropagation computes the average gradient across all samples in a batch. DP-SGD requires something different: `per-sample gradients`, where each sample's gradient contribution is computed separately before clipping and noise addition. This is computationally more expensive because we cannot simply average gradients during the backward pass.

Opacus computes `per-sample gradients` using gradient hooks that intercept and transform the backward computation. For a batch of 256 samples, instead of computing one gradient tensor per parameter, we get 256 gradient tensors per parameter. Each gets clipped to `max_grad_norm`, then summed, noised, and averaged. This increases memory usage proportionally to batch size and slows training by roughly 2-5x compared to standard SGD.

Not all PyTorch layers support `per-sample gradient` computation. `BatchNorm` is incompatible because it computes statistics across the batch dimension, coupling sample gradients together in ways that prevent individual clipping. The gradient for one sample depends on all other samples in the batch, violating the independence assumption DP-SGD requires.

Compatible normalization layers include `GroupNorm` and `LayerNorm`, which normalize within each sample independently. `InstanceNorm` also works because it operates on individual samples. When adapting an existing architecture for DP-SGD, replace `BatchNorm` with `GroupNorm` (using groups equal to the number of channels for channel-wise normalization similar to `BatchNorm`'s effect).

Other incompatible operations include any layer that shares state across samples within a batch. Custom layers that accumulate batch statistics, certain attention mechanisms that mix samples before the final output, and operations that depend on batch size all require modification or replacement. Opacus provides `ModuleValidator.fix()` to attempt automatic fixes for common incompatibilities, but complex architectures may require manual adjustment.

## Privacy Amplification by Subsampling

Rather than fixed-size batches, we use `Poisson subsampling` through Opacus. Each training example is included in a batch independently with probability q=batch_size/dataset_size. This randomness provides `privacy amplification`: because the attacker does not know which samples were included in any given batch, the effective privacy cost per step is reduced.

How much amplification do we get? It depends on the sampling rate q. For our `CIFAR-10` setup with batch size 256 and 50,000 training samples, q=256/50000=0.00512. This low sampling rate significantly amplifies privacy, allowing us to achieve reasonable epsilon values even over many training steps.

Larger batch sizes mean higher sampling rates and less amplification, which might seem counterintuitive. However, larger batches also mean fewer gradient steps per epoch, and the net effect depends on the specific `privacy accountant` calculations. In practice, moderate batch sizes (128-512) often work well for DP-SGD.

What does `Poisson subsampling` look like in practice? With standard batching, we shuffle the dataset once and iterate through fixed-size chunks. With Poisson subsampling, each sample has an independent 0.512% chance of appearing in any given batch. This means batch sizes vary slightly (averaging 256 but sometimes 240, sometimes 270), and the same sample might appear in multiple batches per epoch or skip an epoch entirely. Opacus handles this automatically when you call `make_private_with_epsilon()`.

## Tuning DP-SGD Hyperparameters

Understanding how each hyperparameter affects the privacy-utility tradeoff helps when adapting to other datasets or stricter privacy requirements.

The `clipping threshold` determines how much gradient information survives each step. Too aggressive (say, 0.1) and most gradients get truncated, starving the model of learning signal. Too permissive (say, 10.0) and we waste privacy budget adding noise scaled to rarely-used headroom. A practical calibration approach: run a few epochs without privacy, compute the 75th percentile of gradient norms, and use that value. For typical CNN architectures on `CIFAR-10`, gradient norms fall between 0.5 and 5.0, making `max_grad_norm = 1.0` a reasonable middle ground.

Batch size affects privacy through subsampling amplification. Smaller batches mean lower sampling rates per step, strengthening privacy amplification, but they also require more gradient steps per epoch (each consuming privacy budget). Larger batches weaken amplification but reduce total steps. Going below 64 often hurts convergence because gradient variance becomes too high.

Learning rate for DP-SGD typically matches or slightly undercuts the baseline rate. Since added noise already reduces effective gradient signal, further reduction may not help. If training diverges, lowering the learning rate can stabilize optimization, but start by matching baseline settings. Some practitioners find DP-SGD works better without momentum, though our experiments use momentum successfully.

The number of epochs creates an interesting tradeoff. More epochs allow longer learning but consume more privacy budget, requiring higher noise per step to maintain a target final epsilon. If convergence happens quickly, fewer epochs with proportionally less noise per step often yields better utility at the same privacy level.

Finally, the failure probability `DELTA` should satisfy δ<1/n where n is the training set size. `CIFAR-10` has 50,000 training samples, so δ=10−5=1/100000 comfortably satisfies δ<1/50000. Violating this constraint weakens the privacy guarantee because the failure probability would exceed the probability of any single individual appearing in the dataset.

