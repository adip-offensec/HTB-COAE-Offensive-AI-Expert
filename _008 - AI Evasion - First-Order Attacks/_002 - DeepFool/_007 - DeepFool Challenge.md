
## Objective

Your task is to craft a targeted adversarial example using a DeepFool-style iterative attack. Unlike untargeted attacks that simply cause any misclassification, this challenge requires you to fool the classifier into predicting a specific target class while minimizing the perturbation under an ℓ2 distance constraint.

The constraint is ‖xadv−x‖2≤threshold, where x is the baseline image and xadv is your crafted adversarial example. Both conditions must hold: the predicted class must equal the specified target label (not just any misclassification), and the ℓ2 distance must be at most the threshold.

## Quick Start

All API endpoints expect images in `[0,1]` pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

Code: bash

```bash
export BASE_URL="http://instance_ip:port"
curl -s "$BASE_URL/health"
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded `PNG` files of shape `28x28` and single channel in `[0,1]` after decoding.

### GET /health

Returns service status and configuration parameters.

Code: bash

```bash
curl -s "$BASE_URL/health" | jq
```

Example response:

Code: json

```json
{
  "status": "ok",
  "l2_threshold": 0.75,
  "index": 95,
  "target": 6
}
```

The `l2_threshold` defines the maximum L2 distance allowed. The `index` identifies the MNIST test sample. The `target` specifies the required target class for the adversarial example.

### GET /challenge

Returns the fixed sample and the constraint that must be satisfied.

Code: bash

```bash
curl -s "$BASE_URL/challenge" | jq
```

Example response shape:

Code: json

```json
{
  "sample_index": 95,
  "label": 4,
  "target": 6,
  "l2_threshold": 0.75,
  "image_b64": "<base64 PNG>"
}
```

The `sample_index` identifies the specific MNIST test sample used for this challenge. The `label` is the ground-truth class of the baseline image. The `target` is the required predicted class for a successful adversarial example (targeted attack). The `l2_threshold` defines the maximum $\ell_2$ distance allowed in `[0,1]` pixel space. The `image_b64` field contains the baseline image as a base64-encoded `PNG`.

### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

Code: bash

```bash
curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
```

Example response shape:

Code: json

```json
{
  "pred": 8,
  "confidence": 0.73
}
```

The `pred` field contains the predicted class (integer in `[0,9]`). The `confidence` field provides the softmax probability for the predicted class (float in `[0,1]`).

### GET /weights

Returns the pretrained model `state_dict` as a binary file. Load it on CPU and use the provided classifier architecture to compute input gradients locally.

Code: bash

```bash
curl -s -o deepfool_weights.pth "$BASE_URL/weights"
```

The downloaded file is a PyTorch state dictionary compatible with the `SimpleClassifier` architecture detailed below. Load with `torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit

Validates that the candidate is a proper `28x28` single-channel image in `[0,1]`, that the ℓ2 distance to the baseline image is at most `l2_threshold`, and that the classifier’s predicted class matches the target label. Returns the flag on success.

Code: bash

```bash
curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
```

Example success response shape:

Code: json

```json
{
  "ok": true,
  "pred": 6,
  "target": 6,
  "l2": 0.68,
  "flag": "HTB{...}"
}
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g., `"L2 too large: 0.82 > 0.75"` or `"Wrong target: predicted 8, need 6"`).

## Minimal Python Scaffolds

The following scaffolds provide API helpers, model architecture, and integration testing. They do not implement the attack.

### API Helpers and Challenge Fetching

Code: python

```python
import os, io, base64, numpy as np, requests, torch, torch.nn as nn
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

def l2(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance between two arrays.

    Args:
        a, b: Arrays to compare

    Returns:
        float: Euclidean distance
    """
    return float(np.linalg.norm((a - b).ravel(), ord=2))

# Fetch challenge and check clean prediction
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = x01_from_b64_png(ch["image_b64"])      # (28, 28)
lab = int(ch["label"])                     # baseline label
tgt = int(ch["target"])                    # target label
thr = float(ch["l2_threshold"])            # numeric threshold
res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print({"baseline_label": lab, "target": tgt, "server_pred": res["pred"], "l2_threshold": thr})
```

### Model Architecture and Loading

The server uses this architecture. You need to replicate it locally to compute gradients.

Code: python

```python
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """Forward pass with internal normalization.

        Args:
            x01: Input tensor in [0,1] with shape (N, 1, 28, 28)

        Returns:
            Log-probabilities with shape (N, 10)
        """
        x = (x01 - MNIST_MEAN) / MNIST_STD
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Download and load weights
wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
open("deepfool_weights.pth", "wb").write(wt)

model = SimpleClassifier().eval()
state = torch.load("deepfool_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)

# Verify model works locally
x_tensor = torch.from_numpy(x[None, None, ...]).float()
logits = model(x_tensor)
local_pred = int(torch.argmax(logits, dim=1).item())
print(f"Local prediction: {local_pred}, should match server: {res['pred']}")
```

### Testing Server Validation

Verify server-side checks by submitting the clean image and observing expected failures.

Code: python

```python
bad = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(x)}, timeout=10)
print(bad.status_code, bad.text)  # expected: 400 with "Wrong target: predicted X, need 6"
```

1. After successfully completing the challenge, what is the flag you receive?

```python
import os
import io
import base64
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

BASE_URL = "http://94.237.120.119:33086"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

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

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel(), ord=2))

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - MNIST_MEAN) / MNIST_STD
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Fetch challenge & verify
print("Fetching challenge...")
ch = requests.get(f"{BASE_URL}/challenge", timeout=12).json()
x = x01_from_b64_png(ch["image_b64"])
lab = int(ch["label"])
tgt = int(ch["target"])
thr = float(ch["l2_threshold"])

res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print(f"Baseline: label={lab}, server_pred={res['pred']}, target={tgt}, threshold={thr}")

# Load model
wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
with open("deepfool_weights.pth", "wb") as f:
    f.write(wt)

model = SimpleClassifier().eval()
state = torch.load("deepfool_weights.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)

x_tensor = torch.from_numpy(x[None, None, ...]).float()
local_pred = model(x_tensor).argmax(1).item()
print(f"Local model prediction: {local_pred} (matches server: {local_pred == res['pred']})")

# ───────────────────────────────────────────────
# Corrected momentum-based targeted DeepFool
# ───────────────────────────────────────────────
def deepfool_momentum_targeted(
    x: np.ndarray,
    model: nn.Module,
    target: int,
    max_l2: float,
    max_iter: int = 1500,
    base_overshoot: float = 0.12,
    momentum: float = 0.80,
    max_step_norm: float = 0.07,
    min_step_norm: float = 0.0005,
    confidence_margin: float = 1.5
) -> tuple[np.ndarray, float]:
    x0 = torch.from_numpy(x[None, None, ...]).float()
    x_adv = x0.clone().detach().requires_grad_(True)
    
    velocity = torch.zeros_like(x_adv)
    
    best_adv = x0.clone().detach()
    best_l2 = float('inf')
    best_iter = -1
    
    for i in range(max_iter):
        log_probs = model(x_adv)
        pred = log_probs.argmax(1).item()
        
        current_l2 = torch.norm(x_adv - x0).item()
        
        if pred == target:
            target_lp = log_probs[0, target].item()
            max_other_lp = log_probs[0, [j for j in range(10) if j != target]].max().item()
            margin = target_lp - max_other_lp
            if margin > confidence_margin:
                if current_l2 < best_l2:
                    best_l2 = current_l2
                    best_adv = x_adv.clone().detach()
                    best_iter = i
                    print(f"  [iter {i:4d}] STRONG SUCCESS  L2={current_l2:.5f}  margin={margin:.2f}  conf={torch.exp(torch.tensor(target_lp)).item():.4f}")
            else:
                if current_l2 < best_l2:
                    best_l2 = current_l2
                    best_adv = x_adv.clone().detach()
                    best_iter = i
                    print(f"  [iter {i:4d}] TARGET HIT (weak)  L2={current_l2:.5f}  margin={margin:.2f}")
        
        if best_l2 <= max_l2:
            print(f"→ Best valid adv at iter {best_iter}")
            break
        
        target_lp = log_probs[0, target]
        max_other_lp = log_probs[0, [j for j in range(10) if j != target]].max()
        f = target_lp - max_other_lp
        
        grads = torch.autograd.grad(f, x_adv, retain_graph=False)[0]
        grad_norm = grads.norm(p=2)
        
        if grad_norm < 1e-7:
            print("Gradient vanished → stopping")
            break
        
        # ─── IMPORTANT FIX ───
        # We want to INCREASE f → move in the direction of the gradient
        direction = grads / grad_norm
        
        # Adaptive step size (larger when far from decision boundary)
        step_size = (1.0 + base_overshoot) * (max(1.0, -f.item())) / grad_norm
        
        velocity = momentum * velocity + (1 - momentum) * direction * step_size
        
        v_norm = velocity.norm()
        if v_norm > max_step_norm:
            velocity *= max_step_norm / v_norm
        if v_norm < min_step_norm and v_norm > 0:
            velocity *= min_step_norm / v_norm
        
        x_adv = x_adv + velocity
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        pert = x_adv - x0
        p_norm = pert.norm()
        if p_norm > max_l2:
            x_adv = x0 + max_l2 * pert / p_norm
        
        x_adv = x_adv.detach().requires_grad_(True)
        
        if i % 20 == 0 or i < 100:
            conf = torch.exp(log_probs[0, target]).item()
            print(f"[iter {i:4d}] pred={pred}  f={f.item():+6.3f}  L2={current_l2:.5f}  target_conf={conf:.4f}")
    
    final_l2 = torch.norm(best_adv - x0).item()
    return best_adv.squeeze().cpu().numpy(), final_l2

print("\nRunning corrected momentum-based targeted DeepFool...")
x_adv, final_l2 = deepfool_momentum_targeted(
    x, model, tgt, thr,
    max_iter=1500,
    base_overshoot=0.12,
    momentum=0.80,
    max_step_norm=0.07,
    confidence_margin=1.5
)

print(f"\nFinal stats:")
print(f"  L2 distance     : {final_l2:.5f}  (limit = {thr})")
print(f"  L2 numpy check  : {l2(x_adv, x):.5f}")

x_adv_t = torch.from_numpy(x_adv[None, None, ...]).float()
log_probs_adv = model(x_adv_t)
pred_adv = log_probs_adv.argmax(1).item()
conf_adv = torch.exp(log_probs_adv[0, tgt]).item()

print(f"Local → predicted: {pred_adv}  (target = {tgt})")
print(f"       target confidence: {conf_adv:.4f}")

if pred_adv == tgt and final_l2 <= thr + 1e-5:
    print("\nLooks good — submitting...")
    b64_adv = b64_png_from_x01(x_adv)

    server_res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_adv}, timeout=10).json()
    print(f"Server prediction: {server_res['pred']} (conf {server_res['confidence']:.4f})")

    submit = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_adv}, timeout=12)
    print(f"Submit status: {submit.status_code}")

    if submit.status_code == 200:
        print("SUCCESS!")
        print(submit.json())
    else:
        print("Submit failed:", submit.text)
else:
    print("\nDid not succeed within budget.")
    print("Next ideas if still no flip to class 6:")
    print("  - max_step_norm = 0.09")
    print("  - base_overshoot = 0.18")
    print("  - momentum = 0.70")
    print("  - confidence_margin = 1.0")
    print("  - Or switch to CW-style again with lr=0.015 and class_weight=120")
```

The output is 

```
Fetching challenge...
Baseline: label=4, server_pred=4, target=6, threshold=0.75
Local model prediction: 4 (matches server: True)

Running corrected momentum-based targeted DeepFool...
[iter    0] pred=4  f=-6.335  L2=0.00000  target_conf=0.0017
[iter    1] pred=4  f=-5.615  L2=0.06364  target_conf=0.0035
[iter    2] pred=4  f=-4.978  L2=0.12484  target_conf=0.0065
[iter    3] pred=4  f=-4.370  L2=0.18467  target_conf=0.0116
[iter    4] pred=4  f=-3.783  L2=0.24405  target_conf=0.0201
[iter    5] pred=4  f=-3.239  L2=0.30294  target_conf=0.0329
[iter    6] pred=4  f=-2.703  L2=0.36134  target_conf=0.0525
[iter    7] pred=4  f=-2.206  L2=0.41959  target_conf=0.0794
[iter    8] pred=4  f=-1.729  L2=0.47686  target_conf=0.1156
[iter    9] pred=4  f=-1.319  L2=0.53343  target_conf=0.1572
[iter   10] pred=4  f=-0.969  L2=0.58908  target_conf=0.2015
[iter   11] pred=4  f=-0.585  L2=0.64343  target_conf=0.2616
[iter   12] pred=4  f=-0.191  L2=0.69694  target_conf=0.3344
  [iter   13] TARGET HIT (weak)  L2=0.74971  margin=0.21
→ Best valid adv at iter 13

Final stats:
  L2 distance     : 0.74971  (limit = 0.75)
  L2 numpy check  : 0.74971
Local → predicted: 6  (target = 6)
       target confidence: 0.4185

Looks good — submitting...
Server prediction: 6 (conf 0.4151)
Submit status: 200
SUCCESS!
{'ok': True, 'flag': 'HTB{d33pf00l_f00lz}', 'pred': 6, 'target': 6, 'l2': 0.7487218379974365}
```

