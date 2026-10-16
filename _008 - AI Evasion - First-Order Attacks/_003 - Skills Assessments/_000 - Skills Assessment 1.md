
Your task is to craft a targeted adversarial example using FGSM. Unlike the simple MNIST challenges, this assessment uses CIFAR-10 with color images and a more sophisticated classifier. You must transform a dog image into one the classifier predicts as a cat, while staying within a strict L∞ perturbation budget.

The constraint is ‖xadv−x‖∞≤ϵ where ϵ=8/255 in pixel space. All three conditions must hold: the predicted class equals the target (cat, class 3), the maximum absolute pixel difference is at most ϵ, and all pixel values remain in [0,1]32×32×3.

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
  "challenge": "I-FGSM Targeted Attack on CIFAR-10",
  "sample_index": 42,
  "original_class": 5,
  "original_class_name": "dog",
  "target_class": 3,
  "target_class_name": "cat",
  "epsilon": 0.03137254901960784,
  "epsilon_description": "0.031373 (8/255)",
  "max_iterations_hint": 100,
  "normalization": {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.247, 0.2435, 0.2616]
  },
  "image": "<base64 PNG>",
  "instructions": "Implement targeted I-FGSM..."
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
  "predicted_class": 3,
  "predicted_class_name": "cat",
  "confidence": 0.85,
  "all_probabilities": {
    "airplane": 0.01,
    "automobile": 0.02,
    ...
  }
}
```

### POST /submit

Validates that your candidate is a proper `32×32×3` RGB image in `[0,1]`, that the L∞ distance to the baseline image is at most ϵ, and that the classifier predicts the target class. Returns the flag on success.

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
    "linf_norm": 0.031373,
    "linf_satisfied": true,
    "valid_range": true,
    "adversarial_class": "cat",
    "target_achieved": true
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


def load_model(model_path: str, device: str = "cuda") -> CIFAR10CNN:
    model = CIFAR10CNN(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both direct state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
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

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


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


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.abs(a - b).max())


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
orig_class = int(ch["original_class"])  # 5 (dog)
target_class = int(ch["target_class"])  # 3 (cat)
epsilon = float(ch["epsilon"])  # 8/255
mean = torch.tensor(ch["normalization"]["mean"]).view(3, 1, 1)
std = torch.tensor(ch["normalization"]["std"]).view(3, 1, 1)

# Verify clean prediction
x_norm = (x - mean) / std
with torch.no_grad():
    pred = model(x_norm.unsqueeze(0).to(device)).argmax(dim=1).item()
print(
    f"Original: class {orig_class}, Target: class {target_class}, Clean pred: {pred}, Epsilon: {epsilon:.6f}"
)
```

To verify server-side checks, submit the clean image and observe expected failures:

Code: python

```python
bad = requests.post(
    f"{BASE_URL}/submit",
    json={"image": b64_png_from_tensor(x)},
    timeout=10
)
print(bad.status_code, bad.text)  # expected: success=false, "Target not achieved"
```

1. After successfully completing the assessment task, what is the flag you receive?

```python
import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Model Architecture
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

def load_model(model_path: str, device: str = "cuda") -> CIFAR10CNN:
    model = CIFAR10CNN(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both direct state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model

# Normalization parameters
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.247, 0.2435, 0.2616]

# Helpers
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

# Set BASE_URL
BASE_URL = "http://94.237.53.134:59756"

# Download model weights if not present
weights_path = "cifar10_model_best.pth"
if not os.path.exists(weights_path):
    print("Downloading model weights...")
    resp = requests.get(f"{BASE_URL}/model/weights")
    with open(weights_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {weights_path}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(weights_path, device=device)
print(f"Model loaded on {device}")

# Fetch challenge
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = tensor_from_b64_png(ch["image"])  # (3, 32, 32)
orig_class = int(ch["original_class"])  # 5 (dog)
target_class = int(ch["target_class"])  # 3 (cat)
epsilon = float(ch["epsilon"])  # 8/255
mean = torch.tensor(ch["normalization"]["mean"]).view(3, 1, 1)
std = torch.tensor(ch["normalization"]["std"]).view(3, 1, 1)

# Verify clean prediction
x_norm = (x - mean) / std
with torch.no_grad():
    pred = model(x_norm.unsqueeze(0).to(device)).argmax(dim=1).item()
print(
    f"Original: class {orig_class}, Target: class {target_class}, Clean pred: {pred}, Epsilon: {epsilon:.6f}"
)

# I-FGSM Targeted Attack
x_adv = x.clone().to(device)
original_x = x.to(device)
target = torch.tensor([target_class]).to(device)

alpha = 1 / 255.0
num_iter = 100  # Using the hint for more iterations to ensure success

success = False
for i in range(num_iter):
    x_adv.requires_grad = True
    # Normalize
    x_norm = (x_adv - mean.to(device)) / std.to(device)
    # Forward
    output = model(x_norm.unsqueeze(0))
    # Loss for target class
    loss = F.cross_entropy(output, target)
    # Backward
    model.zero_grad()
    loss.backward()
    # Get gradient
    grad = x_adv.grad.data
    # Update for targeted attack (minimize loss for target)
    x_adv = x_adv - alpha * grad.sign()
    # Clip to epsilon L_inf ball
    x_adv = torch.max(torch.min(x_adv, original_x + epsilon), original_x - epsilon)
    # Clip to [0,1]
    x_adv = torch.clamp(x_adv, min=0, max=1)
    x_adv = x_adv.detach()

    # Check current prediction
    with torch.no_grad():
        x_norm = (x_adv - mean.to(device)) / std.to(device)
        pred_adv = model(x_norm.unsqueeze(0)).argmax(dim=1).item()
        if pred_adv == target_class:
            print(f"Success at iteration {i+1}")
            success = True
            break

if success:
    # Convert to base64
    b64_adv = b64_png_from_tensor(x_adv.cpu())
    
    # Optional: Validate with /predict
    pred_resp = requests.post(
        f"{BASE_URL}/predict",
        json={"image": b64_adv},
        timeout=10
    )
    print("Predict response:", pred_resp.json())
    
    # Submit
    submit_resp = requests.post(
        f"{BASE_URL}/submit",
        json={"image": b64_adv},
        timeout=10
    )
    print("Submit response:", submit_resp.json())
else:
    print("Attack failed after max iterations. Try increasing num_iter or adjusting alpha.")
```

Output is 

```
Downloading model weights...
Saved to cifar10_model_best.pth
/home//htb/Skills_Assessment_1.py:47: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(model_path, map_location=device)
Model loaded on cpu
Original: class 5, Target: class 3, Clean pred: 5, Epsilon: 0.031373
Success at iteration 2
Predict response: {'all_probabilities': {'airplane': 0.0020475767087191343, 'automobile': 0.0005330619751475751, 'bird': 0.025873837992548943, 'cat': 0.4951685667037964, 'deer': 0.015672845765948296, 'dog': 0.39981427788734436, 'frog': 0.04402080550789833, 'horse': 0.011664433404803276, 'ship': 0.0034811925143003464, 'truck': 0.0017234385013580322}, 'confidence': 0.4951685667037964, 'predicted_class': 3, 'predicted_class_name': 'cat'}
Submit response: {'flag': 'HTB{f1r57_0rd3r_fun}', 'message': 'Congratulations! Challenge solved!', 'success': True, 'validation': {'adversarial_class': 'cat', 'adversarial_prediction': 3, 'changed_pixel_ratio': 0.8326823115348816, 'direction_check': True, 'gradient_alignment': 0.8326823115348816, 'linf_constraint': '≤ 0.031373', 'linf_norm': 0.007843166589736938, 'linf_satisfied': True, 'sufficient_coverage': True, 'target_achieved': True, 'target_class': 'cat', 'valid_range': True}}
```

