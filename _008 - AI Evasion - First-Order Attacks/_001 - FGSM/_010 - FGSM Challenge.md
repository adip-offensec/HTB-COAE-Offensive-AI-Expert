
## Objective

Your task is to craft an adversarial example that fools an MNIST digit classifier using the Fast Gradient Sign Method. You will receive a baseline image that the classifier correctly predicts. Your job is to add a small, carefully crafted perturbation that causes misclassification while staying within a strict perturbation budget.

The constraint is ‖xadv−x‖∞≤ϵ, where x is the baseline image and xadv is your crafted adversarial example. Both conditions must hold: the predicted class differs from the baseline label, and the maximum absolute pixel difference is at most `epsilon`.

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
  "epsilon": 0.25,
  "index": 2
}
```

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
  "sample_index": 2,
  "label": 1,
  "epsilon": 0.25,
  "image_b64": "<base64 PNG>"
}
```

The `sample_index` identifies the specific MNIST test sample used for this challenge. The `label` is the ground-truth class. The `epsilon` value defines the maximum L∞ distance allowed in `[0,1]` pixel space. The `image_b64` field contains the baseline image as a base64-encoded `PNG`.

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
curl -s -o fgsm_weights.pth "$BASE_URL/weights"
```

The downloaded file is a PyTorch state dictionary compatible with the `SimpleClassifier` architecture detailed below. Load with `torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit

Validates that the candidate is a proper `28x28` single-channel image in `[0,1]`, that the L∞ distance to the baseline image is at most `epsilon`, and that the classifier’s predicted class differs from the baseline label. Returns the flag on success.

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
  "pred": 8,
  "linf": 0.24,
  "flag": "HTB{...}"
}
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g., `"L_inf too large: 0.26 > 0.25"` or `"Not misclassified; try a stronger perturbation"`).

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

def linf(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L_inf distance between two arrays.

    Args:
        a, b: Arrays to compare

    Returns:
        float: Maximum absolute difference
    """
    return float(np.max(np.abs(a - b)))

# Fetch challenge and check clean prediction
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = x01_from_b64_png(ch["image_b64"])      # (28, 28)
lab = int(ch["label"])                     # baseline label
eps = float(ch["epsilon"])                 # numeric bound
res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print({"baseline_label": lab, "server_pred": res["pred"], "epsilon": eps})
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
open("fgsm_weights.pth", "wb").write(wt)

model = SimpleClassifier().eval()
state = torch.load("fgsm_weights.pth", map_location=torch.device("cpu"))
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
print(bad.status_code, bad.text)  # expected: 400 with "Not misclassified"
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
import torch.nn.functional as F
from PIL import Image
import math

BASE_URL = "http://94.237.59.242:49449"
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

def linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))

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

# Fetch challenge
ch = requests.get(f"{BASE_URL}/challenge").json()
x = x01_from_b64_png(ch["image_b64"])
lab = int(ch["label"])
eps = float(ch["epsilon"])

# Download weights
wt = requests.get(f"{BASE_URL}/weights").content
with open("fgsm_weights.pth", "wb") as f:
    f.write(wt)

# Load model
model = SimpleClassifier().eval()
state = torch.load("fgsm_weights.pth", map_location=torch.device('cpu'))
model.load_state_dict(state)

# Prepare input
x_tensor = torch.from_numpy(x[None, None, ...]).float()
x_tensor.requires_grad = True

# Compute loss and gradient
logits = model(x_tensor)
loss = F.nll_loss(logits, torch.tensor([lab]))

model.zero_grad()
loss.backward()

# Adjust epsilon to avoid rounding issues in PNG quantization
eps_255 = math.floor(eps * 255)
effective_eps = eps_255 / 255.0
print(f"Using effective epsilon: {effective_eps} to ensure L_inf <= {eps} after rounding")

# FGSM perturbation (untargeted)
perturbation = effective_eps * x_tensor.grad.data.sign()
x_adv = torch.clamp(x_tensor + perturbation, min=0, max=1)

# Convert back to numpy
x_adv_np = x_adv.detach().squeeze(0).squeeze(0).numpy()

# Check L_inf
print(f"L_inf distance: {linf(x, x_adv_np)} (should be <= {eps})")

# Local prediction on adv example
logits_adv = model(x_adv)
pred_adv = int(torch.argmax(logits_adv, dim=1).item())
print(f"Local adv prediction: {pred_adv} (should differ from label {lab})")

# Encode adv image
adv_b64 = b64_png_from_x01(x_adv_np)

# Submit to get flag
submit_res = requests.post(f"{BASE_URL}/submit", json={"image_b64": adv_b64}).json()
if submit_res.get("ok"):
    print(f"Flag: {submit_res['flag']}")
else:
    print("Submission failed:", submit_res)
```

We get the following output with the flag

```
$ python3 FGSM_Challange.py 
/home//htb/FGSM_Challange.py:68: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load("fgsm_weights.pth", map_location=torch.device('cpu'))
Using effective epsilon: 0.24705882352941178 to ensure L_inf <= 0.25 after rounding
L_inf distance: 0.24705883860588074 (should be <= 0.25)
Local adv prediction: 8 (should differ from label 1)
Flag: HTB{f457_gr4d13n7_516n_m15l34d5_mn157}
```

