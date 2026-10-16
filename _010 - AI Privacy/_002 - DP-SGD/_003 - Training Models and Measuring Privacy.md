
# Training Models and Measuring Privacy

Before applying DP-SGD, we need to establish a baseline and measure how much membership information the unprotected model leaks. This section trains the non-private baseline, measures its vulnerability, and then trains DP-SGD protected models at different privacy budgets.

## Setting Up the Script

With the imports and helper functions from the Setup Overview in place, begin the demonstration:

Code: python

```python
print("=" * 80)
print("  DP-SGD PRIVACY MITIGATION DEMONSTRATION")
print("=" * 80)
print(f"\nDevice: {device}")
print(f"Random seed: {RANDOM_SEED}")
```

Load the `CIFAR-10` dataset. The first run downloads the data; subsequent runs use the cached version.

Code: python

```python
print("\nLoading CIFAR-10 dataset...")
train_dataset, test_dataset, train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, download=True)

print(f"Training samples: {len(train_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Batch size: {BATCH_SIZE}")
```

## Training the Baseline Model

We train the baseline model with standard SGD (no gradient clipping, no noise injection). This establishes both the accuracy ceiling and the privacy floor.

Code: python

```python
print("\n" + "=" * 80)
print("  TRAINING: BASELINE MODEL (No Privacy Protection)")
print("=" * 80)

baseline_model = CIFAR10CNN().to(device)
baseline_model = train_baseline_sgd(baseline_model, train_loader, device, epochs=BASELINE_EPOCHS, learning_rate=BASELINE_LR)
# Output: Epoch 1/20 - Loss: 2.31, Acc: 25.4%
# Output: Epoch 10/20 - Loss: 1.12, Acc: 61.2%
# Output: Epoch 20/20 - Loss: 0.68, Acc: 77.3%
```

Baseline training runs 20 epochs of standard SGD with momentum 0.9 and learning rate 0.1. Training accuracy rises from about 25% (random guessing across 10 classes) to 75-80%, while loss decreases from around 2.3 (initial cross-entropy) to 0.6-0.8. Each epoch prints progress so you can verify convergence.

## Evaluating Baseline Performance

After training completes, evaluate final performance on both training and test sets:

Code: python

```python
train_acc_baseline = evaluate_accuracy(baseline_model, train_loader, device)
test_acc_baseline = evaluate_accuracy(baseline_model, test_loader, device)

print("\nBaseline Model Performance:")
print(f"  Training accuracy: {train_acc_baseline:.2f}%")
print(f"  Test accuracy: {test_acc_baseline:.2f}%")
print(f"  Overfitting gap: {train_acc_baseline - test_acc_baseline:.2f}%")
```

The overfitting gap (training accuracy minus test accuracy) indicates memorization. A gap of 10% means the model correctly classifies 10% more training samples than test samples, having learned training-specific patterns instead of purely generalizable features.

Save the baseline checkpoint for later comparison and potential submission:

Code: python

```python
torch.save(baseline_model.state_dict(), "output/baseline_model.pth")
print("\nSaved baseline model to output/baseline_model.pth")
```

## Measuring Baseline Membership Inference Vulnerability

Now we measure how much the baseline leaks through its confidence patterns. We use `compute_mia_advantage()` to perform a threshold-based membership inference attack:

Code: python

```python
print("\n" + "=" * 80)
print("  MEMBERSHIP INFERENCE MEASUREMENT: Baseline Model")
print("=" * 80)

mia_acc_baseline, mia_adv_baseline = compute_mia_advantage(
    baseline_model, train_loader, test_loader, device
)

print("\nMIA Results (Baseline):")
print(f"  Attack accuracy: {mia_acc_baseline:.4f}")
print(f"  Attack advantage: {mia_adv_baseline:.4f}")
print("  Random baseline: 0.5000")
```

## Interpreting Baseline Results

In a typical run, training accuracy reaches approximately 77% while test accuracy settles around 67%, producing an overfitting gap of roughly 10 percentage points. The `MIA` advantage typically measures around 0.019 (1.9% above random guessing).

The `MIA` advantage is modest because `CIFAR-10`'s large dataset (50,000 training examples) provides inherent regularization through data diversity. With 5,000 examples per class during training, no single example can dominate the learned parameters.

Even this modest 1.9% advantage means an attacker gains 190 extra correct identifications per 10,000 queries compared to random guessing. For sensitive applications, this systematic leakage justifies privacy protection.

## Defending with DP-SGD

We’ve seen that the baseline leaks membership information through its confidence gap between training and test predictions. DP-SGD addresses this vulnerability at the source by modifying the training procedure itself. We use the Opacus library (covered in the Overview section) to train two DP-SGD models at the privacy budgets discussed earlier: ε=10 and ε=3.

Import the Opacus components needed to convert standard training into DP-SGD:

Code: python

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
```

#### Training with Epsilon = 10

The ε=10 configuration provides modest privacy with manageable accuracy degradation. We create fresh data loaders (Opacus requires its own loader instances) and initialize the model:

Code: python

```python
print("\n" + "=" * 80)
print("  TRAINING: DP-SGD MODEL (Target ε=10)")
print("=" * 80)

TARGET_EPSILON_10 = 10.0

_, _, train_loader_dp, test_loader_dp = get_cifar10_loaders(batch_size=BATCH_SIZE, download=False)

dp_model_10 = CIFAR10CNN().to(device)
dp_model_10 = ModuleValidator.fix(dp_model_10)
optimizer_dp = optim.SGD(dp_model_10.parameters(), lr=DP_LR, momentum=0.9)
```

We use `MAX_GRAD_NORM = 1.0` based on the calibration guidance from the Overview: this value captures the 75th percentile of typical CNN gradient norms on `CIFAR-10`. The `ModuleValidator.fix()` call scans the architecture for incompatible layers and attempts automatic fixes. For architectures containing `BatchNorm`, it would automatically substitute `GroupNorm`. Since `CIFAR10CNN` already uses compatible layers (Conv2d, ReLU, MaxPool2d, Linear), the call returns the model unchanged.

Attach the privacy engine with your target epsilon:

Code: python

```python
privacy_engine = PrivacyEngine(accountant="rdp")
dp_model_10, optimizer_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
    module=dp_model_10,
    optimizer=optimizer_dp,
    data_loader=train_loader_dp,
    target_epsilon=TARGET_EPSILON_10,
    target_delta=DELTA,
    epochs=DP_EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"\nConfiguration:")
print(f"  Target epsilon: {TARGET_EPSILON_10}")
print(f"  Delta: {DELTA}")
print(f"  Max gradient norm: {MAX_GRAD_NORM}")
```

Calling `make_private_with_epsilon()` calculates the `noise multiplier` needed to achieve ε=10 after 20 epochs of training. Internally, Opacus uses the `Rényi differential privacy` (`RDP`) accountant to track privacy loss across multiple gradient steps.

Train and evaluate the model:

Code: python

```python
dp_model_10, final_epsilon_10 = train_dp_sgd(
    dp_model_10, train_loader_dp, privacy_engine, optimizer_dp, device
)
# Output: Epoch 1/20 - Loss: 2.35, Acc: 22.1%, ε: 1.24
# Output: Epoch 10/20 - Loss: 1.52, Acc: 48.3%, ε: 5.82
# Output: Epoch 20/20 - Loss: 1.21, Acc: 58.4%, ε: 10.00

print(f"\nFinal privacy guarantee: (ε={final_epsilon_10:.2f}, δ={DELTA})")
# Output: Final privacy guarantee: (ε=10.00, δ=1e-05)

train_acc_dp10 = evaluate_accuracy(dp_model_10, train_loader_dp, device)
test_acc_dp10 = evaluate_accuracy(dp_model_10, test_loader_dp, device)

print(f"\nDP Model (ε=10) Performance:")
print(f"  Training accuracy: {train_acc_dp10:.2f}%")
print(f"  Test accuracy: {test_acc_dp10:.2f}%")
print(f"  Overfitting gap: {train_acc_dp10 - test_acc_dp10:.2f}%")
# Output: Training accuracy: 61.24%, Test accuracy: 58.15%, Overfitting gap: 3.09%
```

The `privacy_engine` coordinates with the optimizer to clip gradients before each update. During training, gradients exceeding `MAX_GRAD_NORM = 1.0` are scaled down, and Gaussian noise with standard deviation proportional to the `noise multiplier` is added. During these 20 epochs, approximately 60-70% of `per-sample gradients` exceed the clipping threshold and get scaled down. This aggressive clipping bounds sensitivity but slows learning compared to baseline.

Save the model checkpoint. Note the `._module` accessor since Opacus wraps the model to intercept forward passes:

Code: python

```python
torch.save(dp_model_10._module.state_dict(), "output/dp_model_eps10.pth")

# Save in safetensors format for validator API
save_file(dp_model_10._module.state_dict(), "models/dp_model.safetensors")
print("Saved DP model (ε=10) to models/dp_model.safetensors for validation")
```

We use the `safetensors` format because the validation server requires it. We save the ε=10 model specifically because it typically achieves the best balance between accuracy (above 50%) and privacy (`MIA` advantage below 5%).

Measure membership inference on this DP-protected model:

Code: python

```python
print("\n" + "=" * 80)
print("  MEMBERSHIP INFERENCE MEASUREMENT: DP Model (ε=10)")
print("=" * 80)

mia_acc_dp10, mia_adv_dp10 = compute_mia_advantage(
    dp_model_10, train_loader_dp, test_loader_dp, device
)

print(f"\nMIA Results (DP ε=10):")
print(f"  Attack accuracy: {mia_acc_dp10:.4f}")
print(f"  Attack advantage: {mia_adv_dp10:.4f}")

improvement_10 = mia_adv_baseline - mia_adv_dp10
print(f"\nPrivacy Improvement vs Baseline:")
print(f"  MIA advantage reduction: {improvement_10:+.4f}")
print(f"  Accuracy cost: {test_acc_baseline - test_acc_dp10:.2f}%")
```

#### Training with Epsilon = 3

Stronger privacy (lower epsilon) requires more noise, as explained in the Overview. For ε=3, Opacus calculates a `noise multiplier` around 3.8 compared to 1.2 for ε=10. This means roughly 3x more noise per gradient update, making optimization substantially harder.

The training process follows the same pattern as ε=10: fresh data loaders, new model instance, separate privacy engine. The key difference is `TARGET_EPSILON_3 = 3.0`:

Code: python

```python
print("\n" + "=" * 80)
print("  TRAINING: DP-SGD MODEL (Target ε=3)")
print("=" * 80)

TARGET_EPSILON_3 = 3.0

_, _, train_loader_dp3, test_loader_dp3 = get_cifar10_loaders(batch_size=BATCH_SIZE, download=False)

dp_model_3 = CIFAR10CNN().to(device)
dp_model_3 = ModuleValidator.fix(dp_model_3)
optimizer_dp3 = optim.SGD(dp_model_3.parameters(), lr=DP_LR, momentum=0.9)

privacy_engine_3 = PrivacyEngine(accountant="rdp")
dp_model_3, optimizer_dp3, train_loader_dp3 = privacy_engine_3.make_private_with_epsilon(
    module=dp_model_3,
    optimizer=optimizer_dp3,
    data_loader=train_loader_dp3,
    target_epsilon=TARGET_EPSILON_3,
    target_delta=DELTA,
    epochs=DP_EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

dp_model_3, final_epsilon_3 = train_dp_sgd(
    dp_model_3, train_loader_dp3, privacy_engine_3, optimizer_dp3, device
)
# Output: Epoch 20/20 - Loss: 1.45, Acc: 53.2%, ε: 3.00

print(f"\nFinal privacy guarantee: (ε={final_epsilon_3:.2f}, δ={DELTA})")
# Output: Final privacy guarantee: (ε=3.00, δ=1e-05)

torch.save(dp_model_3._module.state_dict(), "output/dp_model_eps3.pth")
```

The higher noise level shows in the results: training converges more slowly, final accuracy drops to around 53%, and the overfitting gap shrinks to roughly 1%. Evaluate and measure `MIA`:

Code: python

```python
train_acc_dp3 = evaluate_accuracy(dp_model_3, train_loader_dp3, device)
test_acc_dp3 = evaluate_accuracy(dp_model_3, test_loader_dp3, device)

print(f"\nDP Model (ε=3) Performance:")
print(f"  Training accuracy: {train_acc_dp3:.2f}%")
print(f"  Test accuracy: {test_acc_dp3:.2f}%")
print(f"  Overfitting gap: {train_acc_dp3 - test_acc_dp3:.2f}%")
# Output: Training accuracy: 54.12%, Test accuracy: 53.01%, Overfitting gap: 1.11%

mia_acc_dp3, mia_adv_dp3 = compute_mia_advantage(
    dp_model_3, train_loader_dp3, test_loader_dp3, device
)

print(f"\nMIA Results (DP ε=3):")
print(f"  Attack accuracy: {mia_acc_dp3:.4f}")
print(f"  Attack advantage: {mia_adv_dp3:.4f}")
# Output: Attack accuracy: 0.5040, Attack advantage: 0.0040

improvement_3 = mia_adv_baseline - mia_adv_dp3
print(f"\nPrivacy Improvement vs Baseline:")
print(f"  MIA advantage reduction: {improvement_3:+.4f}")
print(f"  Accuracy cost: {test_acc_baseline - test_acc_dp3:.2f}%")
# Output: MIA advantage reduction: +0.0150, Accuracy cost: 14.12%
```

## Initial Results Summary

Typical results across the three models:

|Model|Test Accuracy|`MIA` Advantage|Overfitting Gap|
|---|---|---|---|
|Baseline|67%|0.019|10%|
|DP ε=10|58%|0.008|3%|
|DP ε=3|53%|0.004|1%|

Reading across the table, we see the core tradeoff clearly: stronger privacy (lower epsilon) reduces membership advantage but costs accuracy. The baseline achieves 67% test accuracy with a 1.9% `MIA` advantage. Moving to ε=10 drops accuracy by 9 percentage points but cuts `MIA` advantage by more than half (to 0.8%). The stronger ε=3 setting drops accuracy by 14 points total but reduces `MIA` advantage to just 0.4%, barely above random guessing.

The overfitting gap shrinks dramatically with DP-SGD. The baseline memorizes training-specific patterns (10% gap between train and test accuracy), while ε=3 shows almost no memorization (1% gap). This confirms that the noise prevents the model from fitting individual training examples too closely.

The next section generates visualizations and provides detailed analysis of these results.

