DeepFool requires a trained neural network as the attack target. If you completed the FGSM module, the environment setup follows the same pattern using the HTB Evasion Library. The key difference lies in the model architecture: DeepFool benefits from dropout regularization to create more realistic decision boundaries for adversarial testing.

## Environment Setup

The environment setup is identical to the FGSM Setup section, and we will reuse that entire scaffold. The only difference is importing `MNISTClassifierWithDropout` instead of `SimpleCNN`, plus caching utilities (`save_model`, `load_model`, `analyze_model_confidence`). For installation instructions, import details, and explanations of `set_reproducibility`, device selection, and data loading, refer to that section.

Key imports specific to DeepFool:

Code: python

```python
from htb_ai_library import (
    set_reproducibility,
    MNISTClassifierWithDropout,
    get_mnist_loaders,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

set_reproducibility(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Model Architecture: Why Dropout Matters

DeepFool requires a different architecture than FGSM. Why add dropout when FGSM used `SimpleCNN` without it? The answer lies in what we're measuring. An overfitted model might show extremely confident but fragile predictions near training examples, with decision boundaries tightly wrapped around memorized data points. Dropout forces the network to learn redundant representations where multiple feature combinations can indicate the same digit, creating boundaries that better reflect genuine uncertainty.

The architecture uses two convolutional blocks with a 3×3 kernel. The first block outputs 32 channels with 25% dropout, the second outputs 64 channels. Max pooling reduces spatial dimensions from 28×28 to 7×7. After flattening to 3136 dimensions (64×7×7), a fully connected layer with 128 units and 50% dropout processes features before the final 10-class output layer.

The different dropout rates serve specific purposes. Convolutional features are spatially local and numerous (thousands of activations), so conservative 25% dropout prevents destroying spatial information. The fully connected layer has fewer units (128) but each connects to all previous features, creating overfitting risk. The aggressive 50% rate prevents reliance on specific neuron combinations.

During DeepFool's gradient computation, PyTorch automatically disables dropout through `model.eval()`. The attack sees the complete network without random feature zeroing, providing stable gradients for the iterative perturbation process.

### Architecture Definition

The library provides this architecture:

Code: python

```python
# MNISTClassifierWithDropout is imported from htb_ai_library
# The architecture internally defines:
# - Conv1: 1->32 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - Conv2: 32->64 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - FC1: 3136->128, ReLU, 50% dropout
# - FC2: 128->10 (logits)

model = MNISTClassifierWithDropout().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Training with Smart Caching

Training follows the same pattern as FGSM, but with intelligent caching to avoid retraining when a valid model already exists. The pipeline first checks for cached models, validates their accuracy, and only retrains when necessary. This saves significant time during iterative development.

### File Path Configuration

The caching system requires persistent storage for model weights and metadata:

Code: python

```python
model_path = 'output/mnist_model.pth'
os.makedirs('output', exist_ok=True)
```

Why use `output/mnist_model.pth` instead of a temporary file? The `.pth` extension signals PyTorch serialization format to both humans and tools. Storing in a dedicated `output` directory separates generated artifacts from source code, making cleanup and version control exclusion straightforward. The same path serves dual purposes: checking for existing cached models via `os.path.exists(model_path)` and saving newly trained models via `save_model(..., model_path)`.

### Loading and Validating Cached Models

Before training, we check if a cached model exists and whether it meets our accuracy requirements:

Code: python

```python
# Try loading cached model
if os.path.exists(model_path):
    print(f"Found cached model at {model_path}")
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()

    # Validate cached model
    _, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Cached model accuracy: {accuracy:.2f}%")

    if accuracy < 90.0:
        print("Accuracy below threshold, retraining required")
        model = None
else:
    model = None
```

The validation step evaluates the cached model on test data using a batch size of 100, which balances evaluation speed with memory usage. The `evaluate_accuracy` function runs the model on all 10,000 MNIST test samples and computes the percentage of correct predictions. If accuracy falls below 90%, the model is rejected by setting `model = None`, triggering retraining. This threshold ensures we use only well-trained models for adversarial testing. Setting `model.eval()` disables dropout and switches batch normalization to inference mode, ensuring consistent evaluation behavior.

### Training When Required

If no valid cached model exists or the cached model fails validation, training begins:

Code: python

```python
# Train if needed
if model is None:
    print("Training new model...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)
    model = MNISTClassifierWithDropout().to(device)

    model = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
    )
```

Why 5 epochs instead of the single epoch used in FGSM? Dropout regularization slows convergence because each training iteration randomly disables different neurons, preventing the network from memorizing efficient feature paths. A single epoch with dropout typically reaches only 96-97% accuracy. Five epochs allow the network to discover multiple redundant feature combinations and stabilize its decision boundaries, consistently achieving 98-99% accuracy. The training mechanics (forward passes, loss computation, Adam optimization) remain identical to FGSM.

### Persisting Model State

Training produces an ephemeral model in GPU or CPU memory. To enable reuse across sessions, we serialize both weights and metadata:

Code: python

```python
    # Evaluate and cache
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model({
        'model': model,
        'architecture': 'MNISTClassifierWithDropout',
        'accuracy': accuracy,
        'training_config': {
            'epochs': 5,
            'batch_size': 64,
            'device': str(device)
        }
    }, model_path)
```

What gets serialized inside the dictionary? The `model` key stores the complete `nn.Module` object including architecture definition and trained parameters. PyTorch's `torch.save` recursively pickles this object, capturing layer structures, weight tensors, optimizer state, and internal buffers. The remaining keys store human-readable metadata: `architecture` identifies which model class to instantiate when loading, `accuracy` enables the validation step we saw earlier (rejecting models below 90%), and `training_config` preserves hyperparameters for reproducibility. This structure allows loading the model in a fresh Python session without recompiling or retraining. Typical accuracy falls between 98.5-99.0%, meaning approximately 9,850-9,900 correct classifications out of 10,000 test images.

Expected training output:

Code: txt

```txt
Training MNIST classifier on cuda...
Epoch 1/5 - Loss: 0.2134 - Accuracy: 93.54%
Epoch 2/5 - Loss: 0.0891 - Accuracy: 97.23%
Epoch 3/5 - Loss: 0.0623 - Accuracy: 98.12%
Epoch 4/5 - Loss: 0.0502 - Accuracy: 98.54%
Epoch 5/5 - Loss: 0.0421 - Accuracy: 98.76%
Test Accuracy: 98.95%
```

## Model Verification

Verify the trained model exhibits appropriate confidence distributions:

Code: python

```python
# Analyze confidence distribution
_, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
stats = analyze_model_confidence(model, test_loader, device=device, num_samples=1000)
```

The `analyze_model_confidence` function returns a dictionary with confidence statistics. The trained model typically achieves approximately 99% accuracy on correctly predicted samples with high confidence (mean ~0.99), while incorrect predictions show lower confidence (mean ~0.6-0.7). This confidence distribution is ideal for DeepFool: high certainty means finding minimal perturbations requires navigating genuinely learned features rather than exploiting random noise.

