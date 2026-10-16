
Apply your knowledge of sparsity attacks to craft a targeted adversarial example against a ResNet-18 classifier trained on CIFAR-10. You must submit a base64-encoded PNG of shape `32x32x3` that achieves targeted misclassification using either EAD or JSMA. The evaluator validates method signatures and enforces anti-cheating measures including minimum perturbation thresholds.

## Quick start

Begin by verifying the evaluator is ready, then fetch the challenge specification to obtain your sample image, its baseline prediction, and the target class.

  Skills Assessment

```shell-session
@htb[/htb]$ export BASE_URL="http://instance_ip:port"
```

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "ok",
  "model": "...",
  "items": ...
}
```

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/challenge" | jq
{
  "items": [
    {
      "sample_id": ...,
      "label": ...,
      "target": ...,
      "required_method": "...",
      "image_b64": "..."
    }
  ]
}
```

Next, download the model metadata and weights. The architecture is ResNet-18 adapted for CIFAR-10, with the full implementation provided in the scaffolds below.

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/model" | jq
{
  "arch": "ResNetCIFAR",
  "weights_sha256": "...",
  "weights_size": ...,
  "normalize": {
    "mean": [
      ...
    ],
    "std": [
      ...
    ]
  },
  "weights_url": "/model/weights"
}
```

  Skills Assessment

```shell-session
@htb[/htb]$ curl -sL "$BASE_URL/model/weights" -o cifar10_model.pth
```

During development, use the `/predict` endpoint to validate your image encoding and PNG round-trip behavior before final submission.

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 32x32 RGB>"}' | jq
```

## Minimal Python scaffolds

The helpers below provide image I/O and HTTP calls. They are for integration only.

Code: python

```python
import io, os, base64, json, urllib.request
from typing import Dict, Any
from PIL import Image
import numpy as np

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def b64_from_x01(x4d: np.ndarray) -> str:
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def x01_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def http_get(path: str) -> Dict[str, Any]:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


ch = http_get("/challenge")
meta = http_get("/model")
print(
    {
        "items": len(ch["items"]),
        "arch": meta["arch"],
        "weights_url": meta["weights_url"],
    }
)

```

Load the model locally with `ResNetCIFAR` and the downloaded weights.

Code: python

```python
import torch, torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10):
        super().__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def cifar_normalize(x):
    mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=x.dtype, device=x.device)[
        None, :, None, None
    ]
    std = torch.tensor((0.2470, 0.2435, 0.2616), dtype=x.dtype, device=x.device)[
        None, :, None, None
    ]
    return (x - mean) / std


ckpt_path = "cifar10_model.pth"
urllib.request.urlretrieve(f"{BASE_URL}{meta['weights_url']}", ckpt_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCIFAR().to(device).eval()
state = torch.load(ckpt_path, map_location=device)
state_dict = state.get("state_dict_ema") or state.get("state_dict") or state
model.load_state_dict(state_dict)

```

With the model loaded, craft either an EAD or JSMA attack to generate your adversarial example. Before submission, perform a PNG round-trip on your candidate image to mirror the evaluator's decode path and ensure your perturbation survives the encoding process.

## Submission

Submit your adversarial example to the `POST /submit_images` endpoint with the correct `sample_id` and method tag. Images are transmitted as base64-encoded PNGs and automatically decoded to the `[0,1]` range on the server for evaluation.

Code: json

```json
POST /submit_images
{
  "items": [
    { "sample_id": 0, "method": "ead", "image_b64": "..." }
  ]
}
```

The evaluator enforces a minimum L2 perturbation threshold of 1.5 to prevent submitting unmodified clean images. This anti-cheating measure ensures your adversarial example contains actual perturbations rather than the original baseline image. When all validation checks pass (targeted misclassification, method signature, perturbation threshold), the response includes your flag.

1. After successfully completing the task, what is the flag you receive?

```python
import io, os, base64, json, urllib.request
from typing import Dict, Any
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

BASE_URL = os.getenv("BASE_URL", "http://94.237.122.95:49001")

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions (unchanged)
# ────────────────────────────────────────────────────────────────────────────────

def b64_from_x01(x4d: np.ndarray) -> str:
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def x01_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def http_get(path: str) -> Dict[str, Any]:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# ────────────────────────────────────────────────────────────────────────────────
# Model definition (unchanged)
# ────────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def cifar_normalize(x):
    mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=x.dtype, device=x.device)[None, :, None, None]
    std  = torch.tensor((0.2470, 0.2435, 0.2616), dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / std


# ────────────────────────────────────────────────────────────────────────────────
# Load challenge & model
# ────────────────────────────────────────────────────────────────────────────────

ch = http_get("/challenge")
meta = http_get("/model")

ckpt_path = "cifar10_model.pth"
urllib.request.urlretrieve(f"{BASE_URL}{meta['weights_url']}", ckpt_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCIFAR().to(device).eval()

# Recommended: use weights_only=True to suppress the warning
state = torch.load(ckpt_path, map_location=device, weights_only=True)
state_dict = state.get("state_dict_ema") or state.get("state_dict") or state
model.load_state_dict(state_dict)

# ────────────────────────────────────────────────────────────────────────────────
# Get challenge data
# ────────────────────────────────────────────────────────────────────────────────

item = ch["items"][0]
sample_id = item["sample_id"]
true_label = item["label"]
target_label = item["target"]
orig_x = x01_from_b64(item["image_b64"])

print(f"Sample {sample_id} | True: {true_label} → Target: {target_label}")

# ────────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ────────────────────────────────────────────────────────────────────────────────

def get_prediction(x_np: np.ndarray) -> int:
    x_t = torch.from_numpy(x_np).to(device)
    norm_x = cifar_normalize(x_t)
    with torch.no_grad():
        logits = model(norm_x)
    return logits.argmax(-1).item()


assert get_prediction(orig_x) == true_label, "Baseline prediction mismatch"

# ────────────────────────────────────────────────────────────────────────────────
# JSMA Attack with larger step size
# ────────────────────────────────────────────────────────────────────────────────

theta = 0.25          # ← increased from 0.1
max_iter = 4000

x_adv = orig_x.copy().astype(np.float32)

success = False
for it in range(max_iter):
    curr_pred = get_prediction(x_adv)
    if curr_pred == target_label:
        success = True
        print(f"→ Success after {it} iterations")
        break

    # ── Compute gradients ───────────────────────────────────────────────────
    x_torch = torch.from_numpy(x_adv).float().to(device).requires_grad_(True)
    norm_x = cifar_normalize(x_torch)
    logits = model(norm_x)

    grads = []
    for c in range(10):
        x_torch.grad = None
        loss = logits[0, c]
        loss.backward(retain_graph=(c < 9))
        grads.append(x_torch.grad.data.clone())

    grads = torch.stack(grads)  # [10, 3, 32, 32]

    grad_target = grads[target_label]
    grad_other  = grads.sum(dim=0) - grad_target

    # Saliency map: positive for target, negative for others
    mask = (grad_target > 0) & (grad_other < 0)
    saliency = grad_target * torch.abs(grad_other) * mask.float()

    if saliency.max() <= 0:
        print("No valid saliency found → stopping")
        break

    # Pick pixel/feature with highest saliency
    flat_saliency = saliency.view(-1)
    idx = flat_saliency.argmax().item()

    # Direction: increase if grad_target > 0
    pert_sign = torch.sign(grad_target.view(-1)[idx])
    delta = theta * pert_sign.item()

    # Apply perturbation
    x_flat = x_adv.flatten()
    x_flat[idx] += delta
    x_adv = x_flat.reshape(x_adv.shape)

    # Clip
    x_adv = np.clip(x_adv, 0.0, 1.0)

if not success:
    print("Max iterations reached without reaching target class")

# ────────────────────────────────────────────────────────────────────────────────
# Enforce minimum L2 perturbation if needed
# ────────────────────────────────────────────────────────────────────────────────

l2 = np.linalg.norm(x_adv - orig_x)
print(f"Initial L2 perturbation: {l2:.4f}")

MIN_L2 = 1.50

if l2 < MIN_L2:
    print(f"→ Below threshold ({MIN_L2}). Adding extra perturbation...")
    extra_iter = 0
    while l2 < MIN_L2 + 0.05 and extra_iter < 200:
        extra_iter += 1

        x_torch = torch.from_numpy(x_adv).float().to(device).requires_grad_(True)
        norm_x = cifar_normalize(x_torch)
        logits = model(norm_x)

        loss = -logits[0, target_label]  # maximize target logit
        loss.backward()

        grad = x_torch.grad.data
        if grad.abs().max() == 0:
            break

        # Take small step in gradient direction
        step = 0.015 * torch.sign(grad)
        x_adv += step.cpu().numpy()
        x_adv = np.clip(x_adv, 0.0, 1.0)

        l2 = np.linalg.norm(x_adv - orig_x)
        if extra_iter % 20 == 0:
            print(f"  extra {extra_iter:3d} | L2 = {l2:.4f}")

    print(f"Final L2 after enforcement: {l2:.4f}")

# ────────────────────────────────────────────────────────────────────────────────
# PNG round-trip & final check
# ────────────────────────────────────────────────────────────────────────────────

adv_b64 = b64_from_x01(x_adv)
x_round = x01_from_b64(adv_b64)
final_pred = get_prediction(x_round)

print(f"Final prediction after PNG round-trip: {final_pred}")
print("→ SUCCESS" if final_pred == target_label else "→ FAILED")

# ────────────────────────────────────────────────────────────────────────────────
# Submit
# ────────────────────────────────────────────────────────────────────────────────

payload = {
    "items": [
        {
            "sample_id": sample_id,
            "method": "jsma",               # ← important!
            "image_b64": adv_b64
        }
    ]
}

print("Submitting...")
resp = http_post("/submit_images", payload)
print(json.dumps(resp, indent=2))
```

The output is below

```
Sample 0 | True: 5 → Target: 8
→ Success after 121 iterations
Initial L2 perturbation: 2.7875
Final prediction after PNG round-trip: 8
→ SUCCESS
Submitting...
{
  "ok": true,
  "flag": "HTB{5p4r17y_15_pr377y_5p4r53}",
  "decisions": [
    {
      "sample_id": 0,
      "pred": 8,
      "target": 8,
      "method": "either",
      "target_ok": true,
      "signature_ok": true,
      "inferred_method": "ead",
      "delta_l2": 2.798288345336914,
      "delta_linf": 0.5019607543945312
    }
  ],
  "inferred": {
    "0": "ead"
  },
  "diversity": 1
}

```

