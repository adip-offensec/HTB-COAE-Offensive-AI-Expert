
Your organization needs a privacy-preserving digit classifier for processing street view imagery. The model must be trained on the SVHN (Street View House Numbers) dataset using DP-SGD, achieving both acceptable accuracy and strong resistance to membership inference attacks.

## Requirements

Your submission must satisfy both criteria to receive the flag:

|Metric|Requirement|
|---|---|
|`accuracy`|≥ 55%|
|`mia_advantage`|≤ 5%|

## API Endpoints

  DP-SGD Challenge

```shell-session
@htb[/htb]$ export BASE_URL="http://INSTANCE_IP:PORT"
```

#### GET /health

  DP-SGD Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "ok",
  "data_loaded": true
}
```

#### POST /validate

Validates your model and returns the flag on success.

  DP-SGD Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/validate" \
  -F "model=@dp_model.safetensors" | jq
{
  "passed": true,
  "accuracy": 0.5612,
  "mia_advantage": 0.0055,
  "evaluation_time": 4.2,
  "flag": "HTB{...}"
}
```

On failure:

  DP-SGD Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/validate" \
  -F "model=@dp_model.safetensors" | jq
{
  "passed": false,
  "accuracy": 0.5831,
  "mia_advantage": 0.0082,
  "evaluation_time": 3.9
}
```

#### GET /

  DP-SGD Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/" | jq
{
  "service": "DP-SGD Privacy Challenge (SVHN)",
  "version": "1.0"
}
```

## Model Architecture

Your model must use this exact architecture (`BatchNorm` excluded for Opacus compatibility):

Code: python

```python
import torch.nn as nn
import torch.nn.functional as F

class SVHNCNN(nn.Module):
    """CNN for SVHN classification (Opacus compatible)."""

    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Data Preprocessing

The validator uses SVHN-specific normalization:

Code: python

```python
from torchvision import datasets, transforms

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

train_dataset = datasets.SVHN("data", split='train', download=True, transform=transform)
test_dataset = datasets.SVHN("data", split='test', download=True, transform=transform)
```

## Submission Format

Save your trained model using `safetensors`:

Code: python

```python
from safetensors.torch import save_file

# If using Opacus, access model._module.state_dict()
save_file(model._module.state_dict(), "dp_model.safetensors")
```

Submit:

Code: python

```python
import requests

with open("dp_model.safetensors", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/validate",
        files={"model": ("dp_model.safetensors", f, "application/octet-stream")}
    )
print(response.json())
```

1. What is the flag you get for successfully solving the challenge?

Install `pip install torch torchvision opacus safetensors requests` then run below code 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from safetensors.torch import save_file
import warnings

# Suppress some Opacus warnings for cleaner output (common in experiments)
warnings.filterwarnings("ignore", category=UserWarning)

# ========================
# Model Definition (exact as required)
# ========================
class SVHNCNN(nn.Module):
    """CNN for SVHN classification (Opacus compatible - no BatchNorm)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 4 * 4, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========================
# Data Loading & Transform (exact SVHN normalization)
# ========================
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1980, 0.2010, 0.1970)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

train_dataset = datasets.SVHN(root="data", split='train', download=True, transform=transform)
test_dataset  = datasets.SVHN(root="data", split='test',  download=True, transform=transform)

# ========================
# Hyperparameters (tuned for better convergence on this CNN + SVHN)
# ========================
BATCH_SIZE       = 512          # Smaller → lower sampling rate → lower ε for same noise
LR               = 0.05         # Much lower than 0.2 to avoid explosion with noise
EPOCHS           = 20           # DP needs more epochs to converge
MAX_GRAD_NORM    = 1.5          # Tighter clipping helps stability
NOISE_MULTIPLIER = 1.2          # Reduced from 2.75 → better utility, still private
DELTA            = 1e-5

# ========================
# DataLoader
# ========================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,          # Faster loading if you have multiple cores
    pin_memory=True         # Faster to GPU
)

# ========================
# Model, Optimizer, Privacy
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVHNCNN().to(device)

# Validate/fix model for Opacus (should be fine, but good practice)
errors = ModuleValidator.validate(model, strict=False)
if errors:
    print("Model had Opacus compatibility issues; fixing...")
    model = ModuleValidator.fix(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)  # Momentum often helps DP-SGD

# Attach Privacy Engine
privacy_engine = PrivacyEngine()

# Option A: Fixed noise (recommended to start with)
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    poisson_sampling=True,          # Usually better variance
)

# Option B: Target specific epsilon (uncomment if you prefer auto-tuned noise)
# model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     epochs=EPOCHS,
#     target_epsilon=8.0,           # Reasonable target; adjust 4-12 range
#     target_delta=DELTA,
#     max_grad_norm=MAX_GRAD_NORM,
# )

# ========================
# Training Loop
# ========================
print("Starting DP-SGD training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    train_loss /= total
    train_acc = 100. * correct / total
    epsilon = privacy_engine.get_epsilon(DELTA)

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | ε={epsilon:.2f} (δ={DELTA})")

# ========================
# Final Test Evaluation (local check)
# ========================
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        test_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

test_loss /= total
test_acc = 100. * correct / total
print(f"\nFinal Test Evaluation: Loss {test_loss:.4f} | Test Acc {test_acc:.2f}%")
print(f"Privacy: ε ≈ {privacy_engine.get_epsilon(DELTA):.2f} (δ={DELTA})")

# ========================
# Save model for submission
# ========================
save_file(model._module.state_dict(), "dp_model.safetensors")
print("\nModel saved as dp_model.safetensors")
```

Output is below

```
Using downloaded and verified file: data/train_32x32.mat
Using downloaded and verified file: data/test_32x32.mat
Starting DP-SGD training...
Epoch   1/20 | Loss: 2.2355 | Train Acc: 19.20% | ε=0.36 (δ=1e-05)
Epoch   5/20 | Loss: 1.3569 | Train Acc: 65.10% | ε=0.75 (δ=1e-05)
Epoch  10/20 | Loss: 1.3835 | Train Acc: 74.85% | ε=1.06 (δ=1e-05)
Epoch  15/20 | Loss: 1.4647 | Train Acc: 76.44% | ε=1.31 (δ=1e-05)
Epoch  20/20 | Loss: 1.5319 | Train Acc: 77.46% | ε=1.52 (δ=1e-05)

Final Test Evaluation: Loss 1.7293 | Test Acc 76.71%
Privacy: ε ≈ 1.52 (δ=1e-05)
```

Now in a a seperate script validate the `dp_model.safetensors` file

```python
# ========================
# Submission snippet (run separately or here)
# ========================
import requests
BASE_URL = "http://83.136.255.170:50748"  # <-- replace with your challenge URL
with open("dp_model.safetensors", "rb") as f:
     files = {"model": ("dp_model.safetensors", f, "application/octet-stream")}
     response = requests.post(f"{BASE_URL}/validate", files=files)
     print(response.json())
```

Output is below 

```
{'accuracy': 0.7582, 'evaluation_time': 10.12, 'flag': 'HTB{svhn_pr1v4cy_sw33t_sp0t}', 'mia_advantage': 0.015, 'passed': True}
```

