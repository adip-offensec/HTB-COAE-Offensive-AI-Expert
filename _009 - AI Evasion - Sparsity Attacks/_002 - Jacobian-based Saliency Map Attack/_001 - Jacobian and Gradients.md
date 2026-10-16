
# Jacobian and Gradients

To rank pixels by their attack effectiveness, JSMA needs sensitivity information for every class-pixel pair. Computing the full Jacobian matrix requires one backward pass per output class, then extracting target and competitor gradients for saliency scoring. The implementation divides into focused functions: gradient computation, extraction, and masking.

## Setup

To implement JSMA's gradient-based saliency scoring, we need PyTorch for automatic differentiation, NumPy for efficient array operations, and the HTB Evasion Library for model infrastructure. Let's configure the environment with the same standardized setup used throughout this series:

Code: python

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from htb_ai_library.core import set_reproducibility
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import SimpleLeNet
from htb_ai_library.training import train_model
from htb_ai_library.utils import save_model, load_model
from htb_ai_library.visualization import use_htb_style

use_htb_style()
set_reproducibility(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
```

To train and evaluate our attacks, we need properly loaded data and a pre-trained target model. Let's set up both using checkpoint caching to avoid redundant training:

Code: python

```python
train_loader, test_loader = get_mnist_loaders(batch_size=128)
model_path = output_dir / 'mnist_target.pth'

model = SimpleLeNet().to(device)

if model_path.exists():
    print(f"Loading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001, device=device)
    save_model(model, model_path)

model.eval()
print(f"Model ready for attacks")
```

Expected output:

```
Using device: cuda
Loading existing model from output/mnist_target.pth
Model loaded from output/mnist_target.pth
Model ready for attacks
```

With the environment configured and model loaded, we can now implement the Jacobian computation functions.

## Jacobian Computation

Our Jacobian matrix captures how each output class responds to changes in each input pixel. For a model with m classes and input with n pixels, the Jacobian has shape (m,n). Computing this matrix requires m backward passes, one per class, which is the computational bottleneck of JSMA.

To build the complete saliency map, we need per-class gradients showing how each pixel influences each output. This information feeds directly into saliency scoring. Let's start by implementing gradient extraction for a single class:

Code: python

```python
def compute_class_gradient(x, model, class_idx, wrt='logits'):
    x_grad = x.detach().requires_grad_(True)
    logits = model(x_grad)

    if wrt == 'logits':
        scalar = logits[0, class_idx]
    else:
        probs = F.softmax(logits, dim=1)
        scalar = probs[0, class_idx]

    scalar.backward()
    grad = x_grad.grad.detach().cpu().numpy().flatten().copy()

    return grad
```

During the forward pass `model(x_grad)`, PyTorch’s autograd builds a computation graph tracking which operations produced each tensor. We then call `backward()` on a scalar to trigger reverse-mode differentiation. Gradients flow backward through this graph, accumulating at each operation. Why must we use a scalar? Autograd needs a definite starting point (a rank-0 single value, not a tensor). Indexing with `[0, class_idx]` extracts the specific class score we care about, converting the 1×C logit tensor into a single number.

Why flatten and copy? Gradients arrive as a 1×1×28×28 tensor matching the input shape, but saliency scoring operates on vectors. Flattening in C-H-W order produces a length-784 vector where index 352 consistently refers to channel 0, row 12, column 16. We call `.copy()` to create an independent numpy array, severing all ties to PyTorch’s computation graph. Without this copy, the gradient tensor remains a view into GPU memory that PyTorch might overwrite during the next backward pass.

As established earlier, we compute gradients with respect to `logits` (pre-softmax) to preserve independent target and competitor sensitivities for accurate saliency scoring. Let's test this on a simple example:

Code: python

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple 2-class model for testing
model_test = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4, 2)
)
model_test.eval()

# Test input: 1×1×2×2 image
x_test = torch.tensor([[[[0.5, 0.3], [0.2, 0.8]]]], requires_grad=True)

# Compute gradient for class 0
grad_class0 = compute_class_gradient(x_test, model_test, 0, wrt='logits')
print(f"Gradient shape: {grad_class0.shape}")
print(f"Gradient for class 0: {grad_class0}")
```

Output:

Code: txt

```txt
Gradient shape: (4,)
Gradient for class 0: [-0.3841548  -0.4724946   0.04130501 -0.23387289]
```

Looking at this gradient, we see how each of the 4 pixels affects class 0's score. Positive values mean increasing that pixel boosts class 0. Negative values mean increasing it suppresses class 0.

With the single-class gradient function working, we can now build the complete Jacobian by collecting gradients for all classes. Each row represents one class's sensitivity to all input features:

Code: python

```python
def compute_jacobian_matrix(x, model, num_classes=10, wrt='logits'):
    if x.shape[0] != 1:
        raise ValueError("compute_jacobian_matrix expects batch size 1")

    jacobian = []
    for class_idx in range(num_classes):
        grad = compute_class_gradient(x, model, class_idx, wrt)
        jacobian.append(grad)

    return np.asarray(jacobian)
```

Why enforce batch size 1? Passing a batch of 4 images would compute gradients averaged across all 4 (PyTorch's default backward behavior), producing meaningless results for per-sample attacks. Validating `batch_size=1` makes the contract explicit: one image in, one Jacobian out. Without this check, a developer might accidentally pass batched data and debug for hours wondering why saliency maps look wrong.

Building the matrix happens row by row through list accumulation. Each `compute_class_gradient` call triggers a separate backward pass, which for MNIST with 10 classes means 10 independent gradient computations, 10 graph traversals, and 10 memory allocations. This sequential approach is computationally expensive (taking 10-15ms total) but unavoidable (PyTorch requires separate backward passes for each output scalar). Finally, `np.asarray()` converts the list of 1D arrays into a proper (m,n) matrix where row i holds ∂Fi∂xj for all pixels j.

Let's verify the shape on MNIST:

Code: python

```python
# MNIST example: 1×1×28×28 input, 10 classes
# Jacobian should be (10, 784)
print(f"For MNIST: {10} classes × {28*28} pixels = {10 * 28*28} values")
print(f"Expected Jacobian shape: (10, 784)")
```

Output:

Code: txt

```txt
For MNIST: 10 classes × 784 pixels = 7840 values
Expected Jacobian shape: (10, 784)
```

## Gradient Extraction and Masking

With the Jacobian computed, we need to extract the relevant gradients for saliency scoring. The saliency formula requires two components: the target class gradient and the sum of all other class gradients. We separate these components because we want pixels that boost the target while suppressing competitors.

Let's implement these extraction functions:

Code: python

```python
def extract_target_gradient(jacobian, target_class):
    return jacobian[target_class].copy()


def extract_other_gradients(jacobian, target_class):
    target_grad = jacobian[target_class]
    total_grad = jacobian.sum(axis=0)
    other_grad = total_grad - target_grad
    return other_grad
```

Grabbing the target gradient requires indexing into row `target_class` and copying. Why copy instead of returning the view directly? Numpy's row indexing returns a view sharing memory with the original Jacobian. If we modify this view later (say, applying a search space mask), we'd corrupt the Jacobian matrix, causing subsequent extractions to use contaminated gradients. Copying creates an independent array, isolating our modifications from the source data.

Combining all non-target classes into "others" could use manual looping (`for i in range(num_classes): if i != target_class: sum += jacobian[i]`), but array operations run orders of magnitude faster. Summing all rows with `jacobian.sum(axis=0)` gives the total gradient across all classes. Subtracting the target class gradient isolates the combined effect on competitors through simple arithmetic: β=sum(all)−α. For a 3-class problem with target=2, if class 0 has gradient [0.2,0.5], class 1 has [−0.1,0.2], and class 2 has [0.6,−0.3], then the total is [0.2+(−0.1)+0.6,0.5+0.2+(−0.3)]=[0.7,0.4] and others becomes [0.7−0.6,0.4−(−0.3)]=[0.1,0.7].

Example on toy data:

Code: python

```python
# Toy Jacobian: 3 classes, 4 features
J_toy = np.array([
    [ 0.2,  0.5, -0.1,  0.3],  # Class 0
    [-0.1,  0.2,  0.4, -0.2],  # Class 1
    [ 0.6, -0.3,  0.1,  0.5]   # Class 2
])

target = 2
alpha = extract_target_gradient(J_toy, target)
beta = extract_other_gradients(J_toy, target)

print(f"Target gradient (class {target}): {alpha}")
print(f"Other gradients sum:              {beta}")
```

Output:

Code: txt

```txt
Target gradient (class 2): [ 0.6 -0.3  0.1  0.5]
Other gradients sum:       [ 0.1  0.7  0.3  0.1]
```

For pixel 0: target gradient is 0.6 (increasing helps target), other gradient sum is 0.1 (increasing helps competitors slightly). For pixel 1: target gradient is -0.3 (increasing hurts target), other gradient sum is 0.7 (increasing helps competitors a lot). These gradient signs determine which modification directions are favorable, a concept formalized through sign constraints in the saliency scoring functions.

These gradient values feed directly into saliency scoring, but first we need to mask out pixels that are no longer available for modification:

Code: python

```python
def apply_search_mask(gradient, search_space):
    return gradient * search_space
```

How do Boolean masks enable efficient feature filtering? Through element-wise multiplication. The `search_space` array contains `True` for available pixels and `False` for unavailable ones. Numpy's broadcasting converts `True→1.0` and `False→0.0` during multiplication, effectively zeroing gradients for unavailable pixels while preserving others. Why this approach instead of fancy indexing? Masked multiplication preserves array shape and index correspondence. Pixel 352 remains at position 352, just with gradient 0.0 if unavailable. Fancy indexing would create a smaller array of only available pixels, breaking index alignment with the original image.

Example:

Code: python

```python
# Gradient with 5 features
grad = np.array([0.5, -0.2, 0.8, 0.1, -0.4])

# Mask: features 1 and 3 already used
mask = np.array([True, False, True, False, True])

masked_grad = apply_search_mask(grad, mask)
print(f"Original gradient: {grad}")
print(f"Search space mask: {mask}")
print(f"Masked gradient:   {masked_grad}")
```

Output:

Code: txt

```txt
Original gradient: [ 0.5 -0.2  0.8  0.1 -0.4]
Search space mask: [ True False  True False  True]
Masked gradient:   [ 0.5  0.   0.8  0.  -0.4]
```

Features 1 and 3 are zeroed out, preventing them from being selected again.

With these gradient extraction and masking functions complete, we have the necessary inputs for saliency scoring. The next section builds the scoring logic that determines which pixel to modify at each iteration.

