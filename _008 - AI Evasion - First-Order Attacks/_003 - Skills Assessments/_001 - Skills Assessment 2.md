
Your task is to implement the DeepFool algorithm to find a minimal L2 perturbation that causes misclassification on CIFAR-10. This is an untargeted attack where any misclassification succeeds.

The constraint is ‖xadv−xσ‖2≤threshold where the ℓ2 norm is measured in normalized space. All three conditions must hold: the predicted class differs from the original (any class except horse), the ℓ2 distance in normalized space is at most the threshold (3.5), and all pixel values remain in [0,1]32×32×3.

## Quick Start

Start your instance and check readiness:

Code: bash

```bash
export BASE_URL="http://instance_ip:port"
curl -s "$BASE_URL/health"
```

## API

The API is small. Each response is JSON. Images are base64 `PNG` files of shape `32×32×3` (RGB) in `[0,1]` after decoding.

### GET /health

Service health check.

Code: bash

```bash
curl -s "$BASE_URL/health"
```

### GET /challenge

Returns the fixed sample and constraint.

Code: bash

```bash
curl -s "$BASE_URL/challenge" | jq
```

Example response shape:

Code: json

```json
{
  "challenge": "DeepFool Minimal Perturbation on CIFAR-10",
  "sample_index": 137,
  "original_class": 7,
  "original_class_name": "horse",
  "l2_threshold": 3.5,
  "num_classes_hint": 10,
  "overshoot_hint": 0.02,
  "max_iterations_hint": 50,
  "normalization": {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.247, 0.2435, 0.2616]
  },
  "image": "<base64 PNG>",
  "instructions": "Find minimal L2 perturbation..."
}
```

### GET /model/weights

Download the model weights file (`cifar10_model_best.pth`, ~6.3MB). Required for whitebox gradient computation.

Code: bash

```bash
curl -s "$BASE_URL/model/weights" -o cifar10_model_best.pth
```

### POST /predict

Returns the model's prediction for an arbitrary image. Use this to validate your pipeline and intermediate candidates. Does not return the flag.

Code: bash

```bash
curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"image": "<base64 PNG of 32x32 RGB>"}' | jq
```

Example response shape:

Code: json

```json
{
  "predicted_class": 9,
  "predicted_class_name": "truck",
  "confidence": 0.62,
  "all_probabilities": {
    "airplane": 0.01,
    "automobile": 0.02,
    ...
  }
}
```

### POST /submit

Validates that your candidate is a proper `32×32×3` RGB image in `[0,1]`, that the ℓ2 distance to the baseline image in normalized space is at most `l2_threshold`, and that the classifier misclassifies it. Returns the flag on success.

Code: bash

```bash
curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"image": "<base64 PNG of 32x32 RGB>"}' | jq
```

Example success response shape:

Code: json

```json
{
  "success": true,
  "validation": {
    "l2_norm": 0.9632,
    "l2_threshold": 3.5,
    "l2_satisfied": true,
    "valid_range": true,
    "original_class": "horse",
    "adversarial_class": "truck",
    "misclassification": true
  },
  "flag": "HTB{...}"
}
```

## Model Architecture

The target model is a simple CNN trained on CIFAR-10. Save this code as `model.py`:

Code: python

```python
import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path: str, device: str = 'cuda') -> CIFAR10CNN:
    model = CIFAR10CNN(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both direct state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalization parameters (computed from CIFAR-10 training set)
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.247, 0.2435, 0.2616]
```

## Minimal Python Scaffolds

The following helpers fetch the challenge, convert between base64 `PNG` and `[0,1]` tensors, and call the API.

Code: python

```python
import os, io, base64, requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8001")


def tensor_from_b64_png(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    if img.size != (32, 32):
        raise ValueError("Expected 32x32 PNG")
    tensor = transforms.ToTensor()(img)  # converts to (3, 32, 32) in [0,1]
    return tensor


def b64_png_from_tensor(tensor: torch.Tensor) -> str:
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def l2_normalized_space(a: torch.Tensor, b: torch.Tensor, mean, std) -> float:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    a_norm = (a - mean_t) / std_t
    b_norm = (b - mean_t) / std_t
    return float(torch.norm(a_norm - b_norm))


# Download model weights
weights_path = "cifar10_model_best.pth"
if not os.path.exists(weights_path):
    print("Downloading model weights...")
    resp = requests.get(f"{BASE_URL}/model/weights")
    with open(weights_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {weights_path}")

# Load model (assumes model.py from architecture section above)
from model import load_model, NORMALIZATION_MEAN, NORMALIZATION_STD

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(weights_path, device=device)
print(f"Model loaded on {device}")

# Fetch challenge
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = tensor_from_b64_png(ch["image"])  # (3, 32, 32)
orig_class = int(ch["original_class"])  # 7 (horse)
l2_threshold = float(ch["l2_threshold"])  # 3.5
mean = ch["normalization"]["mean"]
std = ch["normalization"]["std"]
mean_t = torch.tensor(mean).view(3, 1, 1)
std_t = torch.tensor(std).view(3, 1, 1)

# Verify clean prediction
x_norm = (x - mean_t) / std_t
with torch.no_grad():
    pred = model(x_norm.unsqueeze(0).to(device)).argmax(dim=1).item()
print(f"Original: class {orig_class}, Clean pred: {pred}, L2 threshold: {l2_threshold}")
```

To verify server-side checks, submit the clean image and observe expected failures:

Code: python

```python
bad = requests.post(
    f"{BASE_URL}/submit",
    json={"image": b64_png_from_tensor(x)},
    timeout=10
)
print(bad.status_code, bad.text)  # expected: success=false, "Misclassification not achieved"
```

1. After successfully completing the assessment task, what is the flag you receive?

```python
import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Model Definition (unchanged)
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1  = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2  = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path: str, device: str = 'cuda') -> CIFAR10CNN:
    model = CIFAR10CNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD  = [0.247,  0.2435, 0.2616]


BASE_URL = "http://83.136.253.5:57257"

def tensor_from_b64_png(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    if img.size != (32, 32):
        raise ValueError("Expected 32x32 PNG")
    return transforms.ToTensor()(img)


def b64_png_from_tensor(tensor: torch.Tensor) -> str:
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def l2_normalized_space(a: torch.Tensor, b: torch.Tensor, mean, std) -> float:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std ).view(3, 1, 1)
    a_norm = (a - mean_t) / std_t
    b_norm = (b - mean_t) / std_t
    return float(torch.norm(a_norm - b_norm, p=2))


# ────────────────────────────────────────────────
# DeepFool – FLIPPED sign to decrease original logit
# ────────────────────────────────────────────────

def deepfool(image_norm: torch.Tensor, net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.005,
             max_iter: int = 150) -> torch.Tensor:
    device = next(net.parameters()).device
    x = image_norm.unsqueeze(0).to(device).clone().detach()

    with torch.no_grad():
        logits_clean = net(x)
        orig_label = logits_clean.argmax(1).item()
        orig_logit_init = logits_clean[0, orig_label].item()
        print(f"Starting – class: {orig_label} | orig_logit: {orig_logit_init:.2f} | all_logits: {logits_clean[0].cpu().numpy().round(2)}")

    pert_x = x.clone().detach()
    prev_orig_logit = orig_logit_init

    for i in range(max_iter):
        pert_x = pert_x.detach().requires_grad_(True)
        logits = net(pert_x)

        current_label = logits.argmax(1).item()
        if current_label != orig_label:
            print(f"→ SUCCESS! Misclassified to {current_label} after {i} iterations")
            break

        f_orig = logits[0, orig_label].item()

        if f_orig > prev_orig_logit + 0.5:  # confidence increasing → wrong direction
            print(f"Warning: orig_logit increasing ({prev_orig_logit:.2f} → {f_orig:.2f}) at iter {i} – likely wrong direction")
            # Optional: break or flip sign dynamically, but for now continue with small steps

        prev_orig_logit = f_orig

        pert_x.grad = None
        logits[0, orig_label].backward(retain_graph=True)
        grad_orig = pert_x.grad.clone()

        min_dist = float('inf')
        r_best = None
        closest_k = -1

        for k in range(num_classes):
            if k == orig_label: continue

            pert_x.grad = None
            retain = (k < num_classes - 1)
            logits[0, k].backward(retain_graph=retain)

            grad_k = pert_x.grad.clone()

            w = grad_k - grad_orig
            norm_w = w.norm(p=2)

            if norm_w < 1e-8: continue

            f_k = logits[0, k].item()
            dist = abs(f_k - f_orig) / norm_w.item()

            # FLIPPED to push DOWN original logit
            r = - (f_k - f_orig) / (norm_w ** 2 + 1e-12) * w

            if dist < min_dist:
                min_dist = dist
                r_best = r
                closest_k = k

        if r_best is None:
            print("No valid direction found – stopping")
            break

        pert_x = pert_x + (1.0 + overshoot) * r_best

        # Clip to reasonable normalized range
        pert_x = pert_x.clamp(-10.0, 10.0)

        current_l2 = (pert_x - x).norm(p=2).item()

        if i % 5 == 0 or i < 15 or current_l2 > 5:
            print(f"Iter {i:3d} | closest={closest_k} | dist≈{min_dist:.4f} | L2={current_l2:.4f} | orig_logit≈{f_orig:.2f} | pred={current_label}")

        if current_l2 > 20:
            print(f"Large L2 ({current_l2:.1f}) at iter {i} – early stop to avoid divergence")
            break

    final_l2 = (pert_x - x).norm(p=2).item()
    print(f"DeepFool finished – final L2 (norm space): {final_l2:.4f}")
    return pert_x.squeeze(0).detach()


# Main (unchanged except call params)
def main():
    weights_path = "cifar10_model_best.pth"
    if not os.path.exists(weights_path):
        print("Downloading model weights...")
        r = requests.get(f"{BASE_URL}/model/weights")
        r.raise_for_status()
        with open(weights_path, "wb") as f:
            f.write(r.content)
        print("Weights saved.")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = load_model(weights_path, device=device_str)
    print(f"Model loaded on {device}")

    print("Fetching challenge...")
    ch = requests.get(f"{BASE_URL}/challenge", timeout=12).json()

    x = tensor_from_b64_png(ch["image"])
    orig_class   = int(ch["original_class"])
    l2_threshold = float(ch["l2_threshold"])
    mean = ch["normalization"]["mean"]
    std  = ch["normalization"]["std"]

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std ).view(3, 1, 1)

    x_norm = (x - mean_t) / std_t

    with torch.no_grad():
        clean_pred = model(x_norm.unsqueeze(0).to(device)).argmax(1).item()
    print(f"Clean prediction: {clean_pred}  (should be {orig_class})")

    print("\nRunning DeepFool (flipped sign + small overshoot + clip)...")
    pert_norm = deepfool(x_norm, model, overshoot=0.005, max_iter=150)

    x_adv = (pert_norm * std_t + mean_t).clamp(0, 1)

    adv_norm = (x_adv - mean_t) / std_t
    with torch.no_grad():
        adv_pred = model(adv_norm.unsqueeze(0).to(device)).argmax(1).item()

    l2_dist = l2_normalized_space(x_adv, x, mean, std)

    print(f"\nFinal adversarial prediction: {adv_pred}")
    print(f"L2 distance (normalized): {l2_dist:.4f} / threshold {l2_threshold}")

    if adv_pred != orig_class and l2_dist <= l2_threshold + 1e-5:
        print("\nLocal success → submitting...")
        b64_adv = b64_png_from_tensor(x_adv)
        resp = requests.post(f"{BASE_URL}/submit", json={"image": b64_adv}, timeout=12)
        print("Server response:")
        print(resp.text)
    else:
        print("\nFailed local check:")
        if adv_pred == orig_class: print("  → still horse")
        if l2_dist > l2_threshold: print(f"  → L2 too large ({l2_dist:.4f})")


if __name__ == "__main__":
    main()
```

Output of the script is

```
Model loaded on cpu
Fetching challenge...
Clean prediction: 7  (should be 7)

Running DeepFool (flipped sign + small overshoot + clip)...
Starting – class: 7 | orig_logit: 2.93 | all_logits: [-2.66 -4.76 -0.92  0.29  1.47  0.74 -3.58  2.93 -5.23 -0.57]
Iter   0 | closest=9 | dist≈0.6724 | L2=0.6757 | orig_logit≈2.93 | pred=7
Iter   1 | closest=9 | dist≈0.1604 | L2=0.8093 | orig_logit≈1.67 | pred=7
→ SUCCESS! Misclassified to 9 after 2 iterations
DeepFool finished – final L2 (norm space): 0.8093

Final adversarial prediction: 9
L2 distance (normalized): 0.8093 / threshold 3.5

Local success → submitting...
Server response:
{"flag":"HTB{d33pf00l_k33p5_0n_f00l1ng}","message":"Congratulations! Challenge solved!","success":true,"validation":{"adversarial_class":"truck","adversarial_prediction":9,"attack_direction":true,"direction_alignment":0.8463354706764221,"l2_norm":0.9594516754150391,"l2_satisfied":true,"l2_threshold":3.5,"linf_norm":0.0156862735748291,"misclassification":true,"original_class":"horse","original_prediction":7,"reasonable_linf":true,"valid_range":true}}
```

