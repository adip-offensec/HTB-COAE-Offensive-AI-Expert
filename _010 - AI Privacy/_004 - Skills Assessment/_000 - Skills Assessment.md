# Skills Assessment

Train a privacy-preserving Fashion-MNIST classifier that resists membership inference attacks while maintaining utility.

## Requirements

Your submission must satisfy both criteria to receive the flag.

Your defended model must maintain at least 70% test accuracy on Fashion-MNIST. This ensures the privacy defense doesn't sacrifice too much utility.

Your defended model must reduce the MIA advantage by at least 40% compared to the vulnerable baseline. The server runs a standardized membership inference attack against your model and compares its vulnerability to the baseline.

Your model must use the `FashionMNISTCNN` architecture exactly as specified below.

## API Endpoints

Set the base URL environment variable to interact with the challenge server.

  Skills Assessment

```shell-session
@htb[/htb]$ export BASE_URL="http://instance_ip:port"
```

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "ok",
  "model_loaded": true
}
```

Use `/baseline` to retrieve information about the vulnerable model's MIA vulnerability. This shows the baseline advantage your model needs to improve upon.

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/baseline" | jq
{
  "baseline_mia_advantage": 0.0895,
  "required_improvement": 0.4,
  "required_test_accuracy": 0.7,
  "description": "The vulnerable baseline model has the MIA advantage shown above. Your defended model must reduce this advantage by at least 40% while maintaining at least 70% test accuracy."
}
```

You can also query the vulnerable model via `/query`.

  Skills Assessment

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{"samples": [[0.1, 0.2, ...], ...]}' | jq
{
  "predictions": [3, 7, ...],
  "confidences": [0.92, 0.87, ...]
}
```

## Dataset and Model

Load Fashion-MNIST with identical normalization to ensure your local samples match the server's preprocessing.

Code: python

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
```

Your defended model must match this architecture exactly to pass validation.

Code: python

```python
import torch.nn as nn


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Submission Format

Save your trained model using safetensors format.

Code: python

```python
from safetensors.torch import save_file

model = FashionMNISTCNN()
# ... train your model with privacy-preserving techniques ...

save_file(model.state_dict(), "defended_model.safetensors")
```

Submit your defended model to the `/submit` endpoint.

  Skills Assessment

```shell-session
@htb[/htb]$ curl -X POST "$BASE_URL/submit" \
    -F "defended_model=@defended_model.safetensors"
```

Code: json

```json
{
  "valid": true,
  "message": "Congratulations! Your model successfully resists membership inference attacks!",
  "flag": "HTB{...}",
  "score": 81.3,
  "details": {
    "test_accuracy": 0.7185,
    "mia_accuracy": 0.5082,
    "mia_advantage": 0.0082,
    "improvement_ratio": 0.9078,
    "baseline_mia_advantage": 0.0895
  }
}
```

1. What is the flag you get for successfully solving the skills assessment?

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from safetensors.torch import save_file
import time
import os

# ───────────────────────────────────────────────────────────────
#  Set BASE_URL directly in the script
# ───────────────────────────────────────────────────────────────
BASE_URL = "http://83.136.255.53:53885"
print(f"BASE_URL set to: {BASE_URL}")

# ───────────────────────────────────────────────────────────────
#  Dataset & Preprocessing (exact match to challenge requirements)
# ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# ───────────────────────────────────────────────────────────────
#  Exact Model Architecture (FashionMNISTCNN as specified)
# ───────────────────────────────────────────────────────────────
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(64 * 3 * 3, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ───────────────────────────────────────────────────────────────
#  Hyperparameters — same as your successful run
# ───────────────────────────────────────────────────────────────
BATCH_SIZE       = 256
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM    = 1.0
LR               = 0.20
MOMENTUM         = 0.9
EPOCHS           = 80
DELTA            = 1e-5

# ───────────────────────────────────────────────────────────────
#  DataLoaders
# ───────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ───────────────────────────────────────────────────────────────
#  Model, Loss, Optimizer, Device
# ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    nesterov=True
)

# ───────────────────────────────────────────────────────────────
#  Attach Opacus Privacy Engine (DP-SGD)
# ───────────────────────────────────────────────────────────────
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    poisson_sampling=True
)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/20)

print(f"Starting DP-SGD training on {device}")
print(f"Batch size: {BATCH_SIZE} | Noise σ: {NOISE_MULTIPLIER} | Clip: {MAX_GRAD_NORM}")
print(f"Target epochs: {EPOCHS} | δ={DELTA}\n")

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    scheduler.step()

    avg_loss = running_loss / total
    train_acc = correct / total
    epsilon = privacy_engine.get_epsilon(DELTA)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | ε={epsilon:.2f} (δ={DELTA})")

# ───────────────────────────────────────────────────────────────
#  Final Test Evaluation
# ───────────────────────────────────────────────────────────────
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

test_acc = correct / total
total_time = (time.time() - start_time) / 60

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final ε ≈ {privacy_engine.get_epsilon(DELTA):.2f}  (δ={DELTA})")
print(f"Total training time: {total_time:.1f} minutes")

# ───────────────────────────────────────────────────────────────
#  Fix: Remove 'module.' prefix added by Opacus before saving
# ───────────────────────────────────────────────────────────────
state_dict = model.state_dict()
clean_state_dict = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}

# Optional: quick check
print("\nSample saved keys:", list(clean_state_dict.keys())[:5])

# ───────────────────────────────────────────────────────────────
#  Save cleaned state_dict
# ───────────────────────────────────────────────────────────────
save_file(clean_state_dict, "defended_model.safetensors")
print("\nModel saved as 'defended_model.safetensors' (with correct parameter names)")

print(f"\nSubmit with:")
print(f'curl -X POST "{BASE_URL}/submit" -F "defended_model=@defended_model.safetensors"')
```

Now run the below script

```python
from safetensors.torch import load_file, save_file

print("Loading current file...")
state = load_file("defended_model.safetensors")

print("Original sample keys:", list(state.keys())[:6])

# Remove the '_module.' prefix
clean_state = {}
for k, v in state.items():
    if k.startswith("_module."):
        new_key = k[len("_module."):]          # strip '_module.' (9 characters)
        clean_state[new_key] = v
    else:
        clean_state[k] = v                      # keep as-is if no prefix

print("\nCleaned sample keys:", list(clean_state.keys())[:6])

# Quick validation
expected = {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias",
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
print("\nAll expected keys present?", expected.issubset(clean_state.keys()))

# Save the fixed version
save_file(clean_state, "defended_model_fixed.safetensors")
print("\nFixed file saved as 'defended_model_fixed.safetensors'")
```

Now upload the fixed file with cURL

```
curl -X POST "http://94.237.61.52:48249/submit" -F "defended_model=@defended_model_fixed.safetensors"
```

Output is

```
{"details":{"baseline_mia_advantage":0.0895,"improvement_ratio":0.9749,"mia_accuracy":0.5022,"mia_advantage":0.0022,"test_accuracy":0.8},"flag":"HTB{pr1v4cy_d3f3nd3r_m4st3r_c0mpl3t3}","message":"Congratulations! Your model successfully resists membership inference attacks!","score":88.74,"valid":true}
```

