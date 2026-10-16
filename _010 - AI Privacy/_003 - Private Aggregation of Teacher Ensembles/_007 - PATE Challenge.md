
Your organization needs a privacy-preserving handwritten letter classifier. The model must be trained using PATE on the EMNIST Letters dataset (26 classes, A-Z), achieving high accuracy while maintaining strong privacy guarantees against membership inference attacks.

## Requirements

Your submission must satisfy both criteria to receive the flag:

|Metric|Requirement|
|---|---|
|`accuracy`|≥ 80%|
|`mia_advantage`|≤ 3%|

## API Endpoints

  PATE Challenge

```shell-session
@htb[/htb]$ export BASE_URL="http://INSTANCE_IP:PORT"
```

#### GET /health

  PATE Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/health" | jq
{
  "status": "ok",
  "data_loaded": true
}
```

#### POST /validate

Validates your student model and returns the flag on success.

  PATE Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/validate" \
  -F "model=@pate_student.safetensors" | jq
{
  "passed": true,
  "accuracy": 0.8612,
  "mia_advantage": 0.0215,
  "evaluation_time": 2.3,
  "flag": "HTB{...}"
}
```

On failure:

  PATE Challenge

```shell-session
@htb[/htb]$ curl -s -X POST "$BASE_URL/validate" \
  -F "model=@pate_student.safetensors" | jq
{
  "passed": false,
  "accuracy": 0.7823,
  "mia_advantage": 0.0183,
  "evaluation_time": 2.1
}
```

#### GET /

  PATE Challenge

```shell-session
@htb[/htb]$ curl -s "$BASE_URL/" | jq
{
  "service": "PATE Privacy Challenge (EMNIST Letters)",
  "version": "1.0"
}
```

## Model Architecture

Your student model must use this exact architecture:

Code: python

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Multi-Layer Perceptron for EMNIST Letters classification."""

    def __init__(self, input_size=784, hidden_layers=None, num_classes=26, dropout=0.2):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output(x)
```

## Data Preprocessing

The validator uses this preprocessing:

Code: python

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

train_dataset = datasets.EMNIST("data", split='letters', train=True, download=True)
test_dataset = datasets.EMNIST("data", split='letters', train=False, download=True)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0

# Labels are 1-26 for A-Z, convert to 0-25
y_train = train_dataset.targets.numpy() - 1
y_test = test_dataset.targets.numpy() - 1

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
```

## Submission Format

Save your trained student model using `safetensors`:

Code: python

```python
from safetensors.torch import save_file

save_file(student_model.state_dict(), "pate_student.safetensors")
```

Submit:

Code: python

```python
import requests

with open("pate_student.safetensors", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/validate",
        files={"model": ("pate_student.safetensors", f, "application/octet-stream")}
    )
print(response.json())
```

1. What is the flag you get for successfully solving the challenge?

Execute the below script.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import save_file
import requests

# -------------------------
# Repro & device
# -------------------------
torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load & preprocess data (MATCHES VALIDATOR)
# -------------------------
train_dataset = datasets.EMNIST("data", split="letters", train=True, download=True)
test_dataset  = datasets.EMNIST("data", split="letters", train=False, download=True)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_dataset.targets.numpy() - 1

X_test  = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test  = test_dataset.targets.numpy() - 1

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------
# Teacher model (free to choose)
# -------------------------
class TeacherMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 26),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Train teachers on disjoint subsets
# -------------------------
num_teachers = 20
epochs_teacher = 12
batch_size_teacher = 128
lr_teacher = 1e-3

perm = np.random.permutation(len(X_train))
splits = np.array_split(perm, num_teachers)

teachers = []
for i in range(num_teachers):
    model = TeacherMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_teacher)
    loss_fn = nn.CrossEntropyLoss()

    subset_idx = splits[i]
    ds = TensorDataset(
        torch.tensor(X_train[subset_idx]),
        torch.tensor(y_train[subset_idx])
    )
    dl = DataLoader(ds, batch_size=batch_size_teacher, shuffle=True)

    model.train()
    for _ in range(epochs_teacher):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    teachers.append(model)

# -------------------------
# PATE aggregation (noisy votes)
# -------------------------
def pate_aggregate(models, data, noise_std=8.0):
    votes = []
    with torch.no_grad():
        for m in models:
            preds = m(torch.tensor(data).to(device)).argmax(dim=1)
            votes.append(preds.cpu().numpy())
    votes = np.stack(votes)  # [T, N]

    labels = []
    for i in range(votes.shape[1]):
        counts = np.bincount(votes[:, i], minlength=26)
        noisy = counts + np.random.normal(0, noise_std, size=26)
        labels.append(np.argmax(noisy))
    return np.array(labels, dtype=np.int64)

# Limit labeled points for better privacy/utility tradeoff
N_STUDENT = 60000
student_labels = pate_aggregate(teachers, X_train[:N_STUDENT], noise_std=5.5)
X_student = X_train[:N_STUDENT]
y_student = student_labels

# -------------------------
# Student model (MUST MATCH EXACT ARCHITECTURE)
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_layers=None, num_classes=26, dropout=0.2):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size
        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output(x)

# -------------------------
# Train student ONLY on noisy labels
# -------------------------
student = MLP().to(device)
opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
loss_s = nn.CrossEntropyLoss()

ds_s = TensorDataset(torch.tensor(X_student), torch.tensor(y_student))
dl_s = DataLoader(ds_s, batch_size=128, shuffle=True)

student.train()
for _ in range(25):
    for xb, yb in dl_s:
        xb, yb = xb.to(device), yb.to(device)
        opt_s.zero_grad()
        loss = loss_s(student(xb), yb)
        loss.backward()
        opt_s.step()

# -------------------------
# Save for submission
# -------------------------
student.eval()
save_file(student.state_dict(), "pate_student.safetensors")
print("Saved pate_student.safetensors")

BASE_URL = "http://94.237.55.124:39229"

print("Submitting...")
with open("pate_student.safetensors", "rb") as f:
    response = requests.post(f"{BASE_URL}/validate", 
                           files={"model": ("pate_student.safetensors", f, "application/octet-stream")})
print(response.json())
```

The output is

```
Saved pate_student.safetensors
Submitting...
{'accuracy': 0.8249, 'evaluation_time': 4.63, 'flag': 'HTB{3mn1st_l3tt3rs_p4t3_m4st3ry}', 'mia_advantage': 0.0112, 'passed': True}
```

