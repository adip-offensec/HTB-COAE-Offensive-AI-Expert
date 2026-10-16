
# Jacobian-based Saliency Map Attack Challenge

Your task is to craft a targeted adversarial example that fools an MNIST digit classifier using the Jacobian-based Saliency Map Attack (JSMA). You will receive a baseline image that the classifier correctly predicts. Your job is to modify a small number of pixels to cause the classifier to predict a specific target class, demonstrating precise, sparse adversarial perturbations.

What you need to do:

Fetch the challenge from the API to receive a baseline `MNIST` image, its ground-truth label, a target class, and an `L0` budget (maximum number of pixels you can modify). Use JSMA to iteratively select and modify the most salient pixels that increase the target class score while decreasing other class scores. Submit your adversarial image to the API. If the classifier predicts the target class and you modified at most the allowed number of pixels, you receive the flag.

The mathematical constraint is an $L_0$ bound on the number of modified pixels:

‖xadv−x‖0≤budget

where x is the baseline image, xadv is your crafted adversarial example, and ‖⋅‖0 counts the number of pixels with non-zero differences (differences above a small threshold like 10−6). The classifier must predict the specified target class t, and all pixel values must remain in [0,1].

## Quick Start

All API endpoints expect images in `[0,1]` pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ export BASE_URL="http://instance_ip:port"
```

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health"
{"status":"healthy"}
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded `PNG` files of shape `28x28` and single channel in `[0,1]` after decoding.

### GET /health

Returns service status and configuration parameters.

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "healthy"
}
```

### GET /challenge

Returns the fixed sample and attack parameters that must be satisfied.

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/challenge" | jq
{
  "sample_index": 2,
  "target_class": 7,
  "l0_budget": 50,
  "original_label": 1,
  "max_l2": 8.0,
  "image_b64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAiUlEQVR42mNgoD9gRhdQe/3uFIzNhC5p+O8pbqPaP+KW0/0yDcFBN1adayVunafuc+OUU/h3gwGnsfYMr3FL6jJ04TTV8u1ZDpw6nYVu/MApqf9/DU5TJV5cR+Gj6EwQO4FbUp7hPe7gefLPCadOW3E8Ed/77ywzLp1cXgxr/uLSyHpsAxfDgAMAUBMg+xetJ/UAAAAASUVORK5CYII="
}
```

The `sample_index` identifies the specific MNIST test sample used for this challenge. The `original_label` is the ground-truth class of the baseline image. The `target_class` is the specific class the adversarial example must be classified as. The `l0_budget` defines the maximum number of pixels that can be modified. The `max_l2` parameter serves as a safeguard to prevent replacing the entire image with unrelated data. The `image_b64` field contains the baseline image as a base64-encoded `PNG`.

### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "predicted_class": 7,
  "confidence": 0.89,
  "all_probabilities": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.89, 0.00, 0.00]
}
```

The `predicted_class` field contains the predicted class (integer in `[0,9]`). The `confidence` field provides the probability for the predicted class (float in `[0,1]`). The `all_probabilities` array contains the probability distribution over all 10 classes.

### GET /weights

Returns the pretrained model `state_dict` as a binary file. Load it on CPU and use the provided classifier architecture to compute the Jacobian matrix locally.

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s -o jsma_weights.pth "$BASE_URL/weights"
```

The downloaded file is a PyTorch state dictionary compatible with the `MNISTClassifier` (LeNet-5 style) architecture detailed below. Load with `torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit

Validates that the candidate is a proper `28x28` single-channel image in [0,1], that the number of modified pixels is at most `l0_budget`, that the L2 distance is below `max_l2`, that the classifier’s predicted class equals the target class, and that the submission is not simply a clean image of the target class copied from the dataset. Returns the flag on success.

  Jacobian-based Saliency Map Attack Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "success": true,
  "flag": "HTB{...}",
  "pixels_modified": 45,
  "predicted_class": 7,
  "target_class": 7
}
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g., `"L0 constraint violated: 55 pixels modified (max: 50)"`, `"Target misclassification failed: predicted 3, expected 7"`, or `"Submission rejected: please derive the adversarial example from the provided baseline image"`).

## Minimal Python Scaffolds

The following scaffolds provide API helpers, model architecture, and integration testing. They do not implement the attack.

### API Helpers and Challenge Fetching

Code: python

```python
import os, io, base64, numpy as np, requests, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def x01_from_b64_png(b64: str) -> np.ndarray:
    """Convert base64 PNG to [0,1] numpy array.

    Args:
        b64: Base64 encoded PNG string

    Returns:
        np.ndarray: Image as (28, 28) array in [0,1] range
    """
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    """Convert [0,1] array to base64 PNG.

    Args:
        x2d: Image array in [0,1] range

    Returns:
        str: Base64 encoded PNG string
    """
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def count_modified_pixels(a: np.ndarray, b: np.ndarray, threshold: float = 1e-6) -> int:
    """Count number of modified pixels (L0 norm).

    Args:
        a, b: Arrays to compare
        threshold: Minimum difference to count as modified

    Returns:
        int: Number of pixels with absolute difference above threshold
    """
    return int(np.sum(np.abs(a - b) > threshold))

# Fetch challenge and check clean prediction
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = x01_from_b64_png(ch["image_b64"])      # (28, 28)
orig_label = int(ch["original_label"])     # baseline label
target = int(ch["target_class"])           # target class for attack
budget = int(ch["l0_budget"])              # max pixels to modify
max_l2 = float(ch["max_l2"])              # L2 safeguard

res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print({"original_label": orig_label, "server_pred": res["predicted_class"],
       "target_class": target, "l0_budget": budget, "max_l2": max_l2})
```

### Model Architecture and Loading

The server uses a LeNet-5 style architecture. You need to replicate it locally to compute the Jacobian matrix.

Code: python

```python
class MNISTClassifier(nn.Module):
    """LeNet-5 style classifier.

    Architecture:
    - Conv1: 1 -> 6 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - Conv2: 6 -> 16 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - FC1: 256 -> 120, Tanh activation
    - FC2: 120 -> 84, Tanh activation
    - FC3: 84 -> 10, log-softmax output
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning log-softmax outputs.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) in [0,1] range

        Returns:
            Log-softmax output of shape (batch, 10)
        """
        x = self.act(self.conv1(x))    # (B,6,24,24)
        x = self.pool(x)               # (B,6,12,12)
        x = self.act(self.conv2(x))    # (B,16,8,8)
        x = self.pool(x)               # (B,16,4,4)
        x = torch.flatten(x, 1)        # (B,256)
        x = self.act(self.fc1(x))      # (B,120)
        x = self.act(self.fc2(x))      # (B,84)
        x = self.fc3(x)                # (B,10)
        return F.log_softmax(x, dim=1)

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] tensor to MNIST statistics.

    Args:
        x01: Input tensor in [0,1] range

    Returns:
        Normalized tensor
    """
    return (x01 - MNIST_MEAN) / MNIST_STD

# Download and load weights
wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
open("jsma_weights.pth", "wb").write(wt)

model = MNISTClassifier().eval()
state = torch.load("jsma_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)

# Verify model works locally
x_tensor = torch.from_numpy(x[None, None, ...]).float()
logits = model(mnist_normalize(x_tensor))
local_pred = int(torch.argmax(logits, dim=1).item())
print(f"Local prediction: {local_pred}, should match server: {res['predicted_class']}")
```

### Testing Server Validation

Verify server-side checks by submitting the clean image and observing expected failures.

Code: python

```python
bad = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(x)}, timeout=10)
print(bad.status_code, bad.text)  # expected: 400 with "Target misclassification failed"
```

1. After successfully completing the task, what is the flag you receive?

```python
import os
import io
import base64
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ======================
# CONFIG
# ======================
BASE_URL = os.getenv("BASE_URL", "http://83.136.249.164:47107")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
THETA = 5.0            # ← much larger now — adjust down if overshooting too fast
MAX_ITER = 300

# ======================
# HELPERS
# ======================
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def count_modified_pixels(a: np.ndarray, b: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(a - b) > threshold))

# ======================
# FETCH CHALLENGE
# ======================
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x_orig = x01_from_b64_png(ch["image_b64"])
orig_label = int(ch["original_label"])
target = int(ch["target_class"])
budget = int(ch["l0_budget"])
max_l2 = float(ch["max_l2"])

print(f"Challenge: {orig_label} → target {target} | Budget: {budget} px | Max L2: {max_l2}")

res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x_orig)}, timeout=10).json()
print(f"Clean server prediction: {res['predicted_class']} (expected {orig_label})")

# ======================
# MODEL
# ======================
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.get_logits(x), dim=1)

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - MNIST_MEAN) / MNIST_STD

# Load weights — safe loading
wt_path = "jsma_weights.pth"
if not os.path.exists(wt_path):
    wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
    open(wt_path, "wb").write(wt)

model = MNISTClassifier().eval()
state = torch.load(wt_path, map_location=torch.device("cpu"), weights_only=True)
model.load_state_dict(state)

# Local check
x_tensor = torch.from_numpy(x_orig[None, None, ...]).float()
local_logits = model.get_logits(mnist_normalize(x_tensor))
local_pred = int(torch.argmax(local_logits, dim=1).item())
print(f"Local clean prediction: {local_pred}")

# ======================
# JSMA CORE
# ======================
def compute_jacobian(model, x_np: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(0)
    x.requires_grad_(True)
    norm = mnist_normalize(x)
    logits = model.get_logits(norm)                  # ← raw logits
    num_pixels = 28 * 28
    jacobian = torch.zeros((num_pixels, 10))

    for i in range(10):
        model.zero_grad()
        logits[0, i].backward(retain_graph=True)
        grad = x.grad.data.view(-1).clone()
        jacobian[:, i] = grad

    # Normalize per class (very helpful for tiny gradients)
    norms = torch.norm(jacobian, dim=0, keepdim=True) + 1e-8
    jacobian = jacobian / norms

    return jacobian

def compute_saliency_map(jacobian: torch.Tensor, target: int, x_flat: torch.Tensor, increase: bool) -> torch.Tensor:
    df_t = jacobian[:, target]
    sum_others = torch.sum(jacobian, dim=1) - df_t

    if increase:
        mask1 = df_t > 0
        mask2 = sum_others < 0
        score = df_t * torch.abs(sum_others)
    else:
        mask1 = df_t < 0
        mask2 = sum_others > 0
        score = torch.abs(df_t) * sum_others

    score = torch.where(mask1 & mask2, score, torch.zeros_like(score))

    boundary_mask = (x_flat < 1.0) if increase else (x_flat > 0.0)
    score *= boundary_mask.float()

    return score

def get_prediction(x_np: np.ndarray) -> int:
    x_t = torch.from_numpy(x_np[None, None, ...]).float()
    with torch.no_grad():
        logits = model.get_logits(mnist_normalize(x_t))
        return int(torch.argmax(logits, dim=1).item())

# ======================
# ATTACK LOOP — SINGLE PIXEL
# ======================
x_adv = np.copy(x_orig).astype(np.float32)
iter_count = 0

while True:
    iter_count += 1
    current_pred = get_prediction(x_adv)
    modified = count_modified_pixels(x_orig, x_adv)

    print(f"[Iter {iter_count:3d}] Pred: {current_pred} | Modified: {modified:2d}/{budget}")

    if current_pred == target:
        print(f"\nSUCCESS → target {target} reached with {modified} pixels changed.")
        break

    if modified >= budget:
        print(f"\nBudget exhausted ({modified} pixels).")
        break

    if iter_count > MAX_ITER:
        print("\nMax iterations reached.")
        break

    J = compute_jacobian(model, x_adv)
    x_flat = torch.from_numpy(x_adv.flatten()).float()

    sal_inc = compute_saliency_map(J, target, x_flat, increase=True)
    sal_dec = compute_saliency_map(J, target, x_flat, increase=False)

    sal_inc_max, p_inc = sal_inc.max(dim=0)
    sal_dec_max, p_dec = sal_dec.max(dim=0)

    print(f"  sal_inc_max = {sal_inc_max:.4e} @ pixel {p_inc.item():4d} | "
          f"sal_dec_max = {sal_dec_max:.4e} @ pixel {p_dec.item():4d}")

    if sal_inc_max < 1e-6 and sal_dec_max < 1e-6:
        print("Saliency too weak after normalization → stopping.")
        break

    if sal_inc_max > sal_dec_max:
        pixel = p_inc.item()
        delta = THETA
    else:
        pixel = p_dec.item()
        delta = -THETA

    r, c = divmod(pixel, 28)
    old_val = x_adv[r, c]
    new_val = np.clip(old_val + delta, 0.0, 1.0)

    if abs(new_val - old_val) < 1e-7:
        print("No meaningful change possible → stalled.")
        break

    x_adv[r, c] = new_val

# ======================
# FINAL CHECK & SUBMIT
# ======================
final_pred = get_prediction(x_adv)
final_modified = count_modified_pixels(x_orig, x_adv)
l2_dist = np.linalg.norm(x_adv - x_orig)

print(f"\nFinal:")
print(f"  Prediction: {final_pred} (wanted {target})")
print(f"  Pixels changed: {final_modified}/{budget}")
print(f"  L2: {l2_dist:.4f} (max {max_l2})")

if final_pred == target and final_modified <= budget and l2_dist <= max_l2 + 0.1:
    print("Requirements met → submitting...")
    resp = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(x_adv)}, timeout=15)
    print(f"Submit status: {resp.status_code}")
    try:
        print(resp.json())
    except:
        print(resp.text)
else:
    print("Did NOT meet requirements.")
```

The output of the prompt is

```
Challenge: 1 → target 7 | Budget: 50 px | Max L2: 8.0
Clean server prediction: 1 (expected 1)
Local clean prediction: 1
[Iter   1] Pred: 1 | Modified:  0/50
  sal_inc_max = 1.1665e-02 @ pixel  202 | sal_dec_max = 1.3950e-03 @ pixel  269
[Iter   2] Pred: 1 | Modified:  1/50
  sal_inc_max = 1.0052e-02 @ pixel  161 | sal_dec_max = 7.3588e-04 @ pixel  268
[Iter   3] Pred: 1 | Modified:  2/50
  sal_inc_max = 7.9234e-03 @ pixel  400 | sal_dec_max = 3.0570e-04 @ pixel  378
[Iter   4] Pred: 1 | Modified:  3/50
  sal_inc_max = 7.3066e-03 @ pixel  427 | sal_dec_max = 1.7063e-03 @ pixel  350
[Iter   5] Pred: 1 | Modified:  4/50
  sal_inc_max = 8.1476e-03 @ pixel  438 | sal_dec_max = 1.5559e-03 @ pixel  350
[Iter   6] Pred: 1 | Modified:  5/50
  sal_inc_max = 8.2112e-03 @ pixel  346 | sal_dec_max = 4.0509e-03 @ pixel  324
[Iter   7] Pred: 1 | Modified:  6/50
  sal_inc_max = 2.3992e-02 @ pixel  411 | sal_dec_max = 4.5292e-03 @ pixel  324
[Iter   8] Pred: 1 | Modified:  7/50
  sal_inc_max = 1.6424e-02 @ pixel  467 | sal_dec_max = 3.6717e-03 @ pixel  324
[Iter   9] Pred: 1 | Modified:  8/50
  sal_inc_max = 5.3154e-03 @ pixel  159 | sal_dec_max = 2.9326e-03 @ pixel  323
[Iter  10] Pred: 1 | Modified:  9/50
  sal_inc_max = 4.3480e-03 @ pixel  148 | sal_dec_max = 2.2126e-03 @ pixel  324
[Iter  11] Pred: 1 | Modified: 10/50
  sal_inc_max = 3.2336e-03 @ pixel  190 | sal_dec_max = 1.9182e-03 @ pixel  324
[Iter  12] Pred: 1 | Modified: 11/50
  sal_inc_max = 2.3682e-03 @ pixel  216 | sal_dec_max = 1.7117e-03 @ pixel  267
[Iter  13] Pred: 1 | Modified: 12/50
  sal_inc_max = 2.0762e-03 @ pixel  188 | sal_dec_max = 1.0515e-03 @ pixel  240
[Iter  14] Pred: 1 | Modified: 13/50
  sal_inc_max = 2.3420e-03 @ pixel  466 | sal_dec_max = 1.0748e-03 @ pixel  240
[Iter  15] Pred: 1 | Modified: 14/50
  sal_inc_max = 3.0271e-03 @ pixel  289 | sal_dec_max = 9.8910e-04 @ pixel  240
[Iter  16] Pred: 7 | Modified: 15/50

SUCCESS → target 7 reached with 15 pixels changed.

Final:
  Prediction: 7 (wanted 7)
  Pixels changed: 15/50
  L2: 3.8730 (max 8.0)
Requirements met → submitting...
Submit status: 200
{'success': True, 'flag': 'HTB{j5m4_l4b_l3n3t_l3c00l}', 'pixels_modified': 15, 'predicted_class': 7, 'target_class': 7}
```

