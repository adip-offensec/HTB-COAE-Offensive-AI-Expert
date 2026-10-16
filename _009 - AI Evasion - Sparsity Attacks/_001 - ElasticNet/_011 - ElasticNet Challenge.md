Your task is to craft an adversarial example that fools an MNIST digit classifier using the ElasticNet (EAD) attack. You will receive a baseline image that the classifier correctly predicts. Your job is to add a carefully crafted perturbation that causes misclassification while respecting multiple distance constraints simultaneously.

What you need to do:

Fetch the challenge from the API to receive a baseline `MNIST` image, its ground-truth label, and the constraint parameters. Use the ElasticNet attack (FISTA optimization with elastic-net regularization and binary search) to modify the image. Ensure your perturbation satisfies all three distance bounds: the elastic-net distance (combining `L2` and `L1` norms), an `L2` cap, and an `L1` cap. Submit your adversarial image to the API. If the classifier misclassifies it and all constraints are satisfied, you receive the flag.

To craft a valid adversarial example, xadv, from a baseline image, x, several mathematical constraints must be met simultaneously. The perturbation is bounded by three conditions: an `elastic-net constraint` (‖xadv−x‖2+β‖xadv−x‖1≤elastic_max), an `auxiliary L2 norm constraint` (‖xadv−x‖2≤l2_max), and an `auxiliary L1 norm constraint` (‖xadv−x‖1≤l1_max), where β is the elastic-net trade-off parameter. Furthermore, for the attack to be successful, the predicted class of the adversarial example must differ from the baseline label, and all its pixel values must remain valid by being clipped within the range [0,1].

## Quick Start

All API endpoints expect images in `[0,1]` pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ export BASE_URL="http://instance_ip:port"
```

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health"
{"status":"ok"}
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded `PNG` files of shape `28x28` and single channel in `[0,1]` after decoding.

### GET /health

Returns service status and configuration parameters.

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "ok"
}
```

### GET /challenge

Returns the fixed sample and all constraints that must be satisfied.

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/challenge" | jq
{
  "sample_index": ...,
  "label": ...,
  "beta": ...,
  "elastic_max": ...,
  "l2_max": ...,
  "l1_max": ...,
  "image_b64": "..."
}
```

The `sample_index` identifies the specific MNIST test sample used for this challenge. The `label` is the ground-truth class. The `beta` parameter controls the elastic-net trade-off between L2 and L1 regularization. The three distance bounds (`elastic_max`, `l2_max`, `l1_max`) define the maximum allowed distortion in `[0,1]` pixel space. The `image_b64` field contains the baseline image as a base64-encoded `PNG`.

### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "pred": ...,
  "confidence": ...
}
```

The `pred` field contains the predicted class (integer in `[0,9]`). The `confidence` field provides the softmax probability for the predicted class (float in `[0,1]`).

### GET /weights

Returns the pretrained model `state_dict` as a binary file. Load it on CPU and use the provided classifier architecture to compute input gradients locally.

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s -o elasticnet_weights.pth "$BASE_URL/weights"
```

The downloaded file is a PyTorch state dictionary compatible with the `SimpleClassifier` architecture detailed below. Load with `torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit

Validates that the candidate is a proper `28x28` single-channel image in `[0,1]`, that all three distance constraints are satisfied (elastic-net distance, L2 bound, and L1 bound), and that the classifier’s predicted class differs from the baseline label. Returns the flag on success.

  ElasticNet Attack Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "ok": true,
  "flag": "HTB{...}",
  "pred": ...,
  "metrics": {
    "l1": ...,
    "l2": ...,
    "linf": ...,
    "elastic": ...
  }
}
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g., `"Elastic too large: 1.85 > 1.80"`, `"L2 too large: 1.42 > 1.40"`, or `"Not misclassified; try stronger within bounds"`).

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

def compute_distances(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute L1, L2, Linf, and elastic-net distances.

    Args:
        a, b: Arrays to compare

    Returns:
        dict: Dictionary containing l1, l2, linf, and elastic distance values
    """
    diff = a - b
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff**2)))
    linf = float(np.max(np.abs(diff)))
    return {"l1": l1, "l2": l2, "linf": linf}

# Fetch challenge and check clean prediction
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = x01_from_b64_png(ch["image_b64"])      # (28, 28)
lab = int(ch["label"])                     # baseline label
beta = float(ch["beta"])                   # elastic-net trade-off
elastic_max = float(ch["elastic_max"])     # elastic-net bound
l2_max = float(ch["l2_max"])              # L2 auxiliary bound
l1_max = float(ch["l1_max"])              # L1 auxiliary bound

res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print({"baseline_label": lab, "server_pred": res["pred"],
       "beta": beta, "elastic_max": elastic_max, "l2_max": l2_max, "l1_max": l1_max})
```

### Model Architecture and Loading

The server uses this architecture. You need to replicate it locally to compute gradients and run FISTA.

Code: python

```python
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without normalization.

        Args:
            x: Input tensor (already normalized) with shape (N, 1, 28, 28)

        Returns:
            Logits with shape (N, 10)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] tensor to MNIST statistics.

    Args:
        x01: Input tensor in [0,1] range with shape (N, 1, 28, 28)

    Returns:
        Normalized tensor
    """
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected (N,1,28,28), got {x01.shape}")
    if x01.min() < 0.0 or x01.max() > 1.0:
        raise ValueError("Input must be in [0,1]")
    return (x01 - MNIST_MEAN) / MNIST_STD

# Download and load weights
wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
open("elasticnet_weights.pth", "wb").write(wt)

model = SimpleClassifier().eval()
state = torch.load("elasticnet_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)

# Verify model works locally
x_tensor = torch.from_numpy(x[None, None, ...]).float()
logits = model(mnist_normalize(x_tensor))
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

1. After successfully completing the task, what is the flag you receive?

Run the following script

```python
import os, io, base64, numpy as np, requests, torch, torch.nn as nn
from PIL import Image
from math import sqrt

BASE_URL = os.getenv("BASE_URL", "http://94.237.122.95:47294")
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

def compute_distances(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute L1, L2, Linf, and elastic-net distances.

    Args:
        a, b: Arrays to compare

    Returns:
        dict: Dictionary containing l1, l2, linf, and elastic distance values
    """
    diff = a - b
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff**2)))
    linf = float(np.max(np.abs(diff)))
    return {"l1": l1, "l2": l2, "linf": linf}

# Fetch challenge and check clean prediction
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = x01_from_b64_png(ch["image_b64"])      # (28, 28)
lab = int(ch["label"])                     # baseline label
beta = float(ch["beta"])                   # elastic-net trade-off
elastic_max = float(ch["elastic_max"])     # elastic-net bound
l2_max = float(ch["l2_max"])              # L2 auxiliary bound
l1_max = float(ch["l1_max"])              # L1 auxiliary bound

res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x)}, timeout=10).json()
print({"baseline_label": lab, "server_pred": res["pred"],
       "beta": beta, "elastic_max": elastic_max, "l2_max": l2_max, "l1_max": l1_max})

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without normalization.

        Args:
            x: Input tensor (already normalized) with shape (N, 1, 28, 28)

        Returns:
            Logits with shape (N, 10)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] tensor to MNIST statistics.

    Args:
        x01: Input tensor in [0,1] range with shape (N, 1, 28, 28)

    Returns:
        Normalized tensor
    """
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected (N,1,28,28), got {x01.shape}")
    if x01.min() < 0.0 or x01.max() > 1.0:
        raise ValueError("Input must be in [0,1]")
    return (x01 - MNIST_MEAN) / MNIST_STD

# Download and load weights
wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
open("elasticnet_weights.pth", "wb").write(wt)

model = SimpleClassifier().eval()
state = torch.load("elasticnet_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)

# Verify model works locally
x_tensor = torch.from_numpy(x[None, None, ...]).float()
logits = model(mnist_normalize(x_tensor))
local_pred = int(torch.argmax(logits, dim=1).item())
print(f"Local prediction: {local_pred}, should match server: {res['pred']}")

# Define helper functions for attack
def get_pred(model, xadv):
    with torch.no_grad():
        logits = model(mnist_normalize(xadv))
        return torch.argmax(logits, dim=1).item()

def soft_threshold(z, thresh):
    return torch.sign(z) * torch.max(torch.abs(z) - thresh, torch.zeros_like(z))

def run_fista(c, beta, step, max_iter, kappa, x_tensor, lab, model):
    delta = torch.zeros_like(x_tensor)
    y = delta.clone()
    t = 1.0
    for k in range(max_iter):
        y.requires_grad = True
        xadv = torch.clamp(x_tensor + y, 0, 1)
        logits = model(mnist_normalize(xadv))[0]
        label_idx = torch.tensor(lab)
        other_mask = torch.arange(10) != label_idx
        f = logits[lab] - torch.max(logits[other_mask])
        loss = c * torch.max(f + kappa, torch.tensor(0.0)) + 0.5 * torch.sum(y ** 2)
        model.zero_grad()
        loss.backward()
        grad = y.grad.clone()
        y.requires_grad = False
        y = y.detach()
        z = y - step * grad
        delta_new = soft_threshold(z, step * beta)
        t_new = (1 + sqrt(1 + 4 * t ** 2)) / 2
        y = delta_new + ((t - 1) / t_new) * (delta_new - delta)
        t = t_new
        delta = delta_new
        # Early stop check
        curr_xadv = torch.clamp(x_tensor + delta, 0, 1)
        curr_pred = get_pred(model, curr_xadv)
        if curr_pred != lab:
            with torch.no_grad():
                curr_logits = model(mnist_normalize(curr_xadv))[0]
                curr_f = curr_logits[lab] - torch.max(curr_logits[other_mask])
            if curr_f <= -kappa:
                break
    return delta

# Attack parameters
kappa = 0.0
step = 0.001
max_iter = 1000
binary_search_steps = 20

# Find an initial high c that succeeds by doubling
c = 0.001
delta = None
while True:
    print(f"Trying c = {c}")
    delta = run_fista(c, beta, step, max_iter, kappa, x_tensor, lab, model)
    xadv = torch.clamp(x_tensor + delta, 0, 1)
    pred = get_pred(model, xadv)
    if pred != lab:
        break
    c *= 2

high = c
low = 0.0
best_delta = delta  # Start with the successful one

for bs_step in range(binary_search_steps):
    mid = (low + high) / 2
    print(f"Binary search step {bs_step}, c = {mid}")
    delta = run_fista(mid, beta, step, max_iter, kappa, x_tensor, lab, model)
    xadv = torch.clamp(x_tensor + delta, 0, 1)
    pred = get_pred(model, xadv)
    if pred != lab:
        best_delta = delta
        high = mid
    else:
        low = mid

# Use the best delta
delta = best_delta
xadv = torch.clamp(x_tensor + delta, 0, 1)
adv_np = xadv[0, 0].numpy()
pred = get_pred(model, xadv)
print(f"Adversarial prediction: {pred}")

# Compute distances
dist = compute_distances(adv_np, x)
elastic = dist["l2"] + beta * dist["l1"]
print(f"Distances: l1={dist['l1']:.4f} (<= {l1_max}), l2={dist['l2']:.4f} (<= {l2_max}), elastic={elastic:.4f} (<= {elastic_max})")

# Submit
submit_res = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(adv_np)}, timeout=10)
print(submit_res.status_code, submit_res.text)

# Testing server validation (optional, for clean image)
bad = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(x)}, timeout=10)
print(bad.status_code, bad.text)  # expected: 400 with "Not misclassified"
```

The output is below

```
{'baseline_label': 9, 'server_pred': 9, 'beta': 0.01, 'elastic_max': 1.8, 'l2_max': 1.4, 'l1_max': 80.0}
/home//htb/ElasticNet_Challenge.py:116: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load("elasticnet_weights.pth", map_location=torch.device("cpu"))
Local prediction: 9, should match server: 9
Trying c = 0.001
Trying c = 0.002
Trying c = 0.004
Trying c = 0.008
Trying c = 0.016
Trying c = 0.032
Trying c = 0.064
Trying c = 0.128
Binary search step 0, c = 0.064
Binary search step 1, c = 0.096
Binary search step 2, c = 0.08
Binary search step 3, c = 0.088
Binary search step 4, c = 0.08399999999999999
Binary search step 5, c = 0.08199999999999999
Binary search step 6, c = 0.08299999999999999
Binary search step 7, c = 0.08249999999999999
Binary search step 8, c = 0.08274999999999999
Binary search step 9, c = 0.08262499999999999
Binary search step 10, c = 0.0826875
Binary search step 11, c = 0.08265624999999999
Binary search step 12, c = 0.082640625
Binary search step 13, c = 0.08263281249999999
Binary search step 14, c = 0.08262890625
Binary search step 15, c = 0.08262695312499999
Binary search step 16, c = 0.08262597656249998
Binary search step 17, c = 0.08262548828124999
Binary search step 18, c = 0.082625244140625
Binary search step 19, c = 0.0826251220703125
Adversarial prediction: 8
Distances: l1=7.0542 (<= 80.0), l2=0.7598 (<= 1.4), elastic=0.8304 (<= 1.8)
200 {"ok":true,"flag":"HTB{3l4st1cn3t_3l4st1c1s1ng_3l4st1c}","pred":8,"metrics":{"l1":7.062745094299316,"l2":0.7615823745727539,"linf":0.23529411852359772,"elastic":0.8322098255157471}}
400 {"detail":"Not misclassified; try stronger within bounds"}
```

