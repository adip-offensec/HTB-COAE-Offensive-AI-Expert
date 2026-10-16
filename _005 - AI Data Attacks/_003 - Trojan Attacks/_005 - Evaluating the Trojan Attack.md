

The final step is evaluating the impact of the actual Trojan attack. We load the saved model weights if necessary and then perform two key evaluations: First, we measure the accuracy of both the clean and trojaned models on the clean test data (`testloader_clean`). High accuracy for the trojaned model here demonstrates the attack's stealth. Second, we calculate the Attack Success Rate (ASR) for both models using the triggered test data (`testloader_triggered`). A high ASR for the trojaned model, coupled with a low ASR for the clean model, confirms the attack's effectiveness, that the backdoor was successfully implanted and activates when the trigger is present.

Code: python

```python
# Initialize variables to store evaluation results
clean_acc_clean_gtsrb = 0.0
clean_asr_gtsrb = 0.0
trojan_acc_clean_gtsrb = 0.0
trojan_asr_gtsrb = 0.0

# Check if model variables exist and if saved files exist (for loading if needed)
clean_model_available = "clean_model_gtsrb" in locals()
trojan_model_available = "trojaned_model_gtsrb" in locals()
clean_model_file_exists = os.path.exists("gtsrb_cnn_clean.pth")
trojan_model_file_exists = os.path.exists("gtsrb_cnn_trojaned.pth")

# Check if necessary dataloaders are available
testloader_clean_available = (
    "testloader_clean" in locals() and testloader_clean is not None
)
testloader_triggered_available = (
    "testloader_triggered" in locals() and testloader_triggered is not None
)

print("\n-- Evaluating Clean GTSRB Model (Baseline) --")
# Load clean model if not already in memory but file exists
if not clean_model_available and clean_model_file_exists:
    print("Loading pre-trained clean model state from gtsrb_cnn_clean.pth...")
    try:
        clean_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
        clean_model_gtsrb.load_state_dict(
            torch.load("gtsrb_cnn_clean.pth", map_location=device)
        )
        clean_model_available = True
        print("Clean model loaded successfully.")
    except Exception as e:
        print(f"Error loading clean model state dict: {e}")
        clean_model_available = False  # Ensure flag is false if loading failed

# Proceed with evaluation only if model and loaders are ready
if clean_model_available and testloader_clean_available:
    # Evaluate accuracy on clean test data
    clean_acc_clean_gtsrb, _, _, _ = evaluate_model(
        clean_model_gtsrb,
        testloader_clean,
        criterion_gtsrb,  # Assumes criterion is still defined
        device,
        description="Clean Model on Clean GTSRB Test Data",
    )
    # Evaluate ASR on triggered test data
    if testloader_triggered_available:
        clean_asr_gtsrb = calculate_asr_gtsrb(
            clean_model_gtsrb,
            testloader_triggered,
            SOURCE_CLASS,
            TARGET_CLASS,
            device,
        )
    else:
        print("Skipping clean model ASR calculation: Triggered testloader unavailable.")
else:
    if not clean_model_available:
        print("Skipping clean model evaluation: Model not available.")
    if not testloader_clean_available:
        print("Skipping clean model evaluation: Clean testloader unavailable.")


print("\n-- Evaluating Trojaned GTSRB Model --")
# Load trojaned model if not already in memory but file exists
if not trojan_model_available and trojan_model_file_exists:
    print("Loading pre-trained trojaned model state from gtsrb_cnn_trojaned.pth...")
    try:
        trojaned_model_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
        trojaned_model_gtsrb.load_state_dict(
            torch.load("gtsrb_cnn_trojaned.pth", map_location=device)
        )
        trojan_model_available = True
        print("Trojaned model loaded successfully.")
    except Exception as e:
        print(f"Error loading trojaned model state dict: {e}")
        trojan_model_available = False

# Proceed with evaluation only if model and loaders are ready
if trojan_model_available and testloader_clean_available:
    # Evaluate accuracy on clean test data (Stealth Check)
    trojan_acc_clean_gtsrb, _, _, _ = evaluate_model(
        trojaned_model_gtsrb,
        testloader_clean,
        criterion_gtsrb,
        device,
        description="Trojaned Model on Clean GTSRB Test Data",
    )
    # Evaluate ASR on triggered test data (Effectiveness Check)
    if testloader_triggered_available:
        trojan_asr_gtsrb = calculate_asr_gtsrb(
            trojaned_model_gtsrb,
            testloader_triggered,
            SOURCE_CLASS,
            TARGET_CLASS,
            device,
        )
    else:
        print(
            "Skipping trojaned model ASR calculation: Triggered testloader unavailable."
        )
else:
    if not trojan_model_available:
        print("Skipping trojaned model evaluation: Model not available.")
    if not testloader_clean_available:
        print("Skipping trojaned model evaluation: Clean testloader unavailable.")
```

We can see the impact the attack has had on the model extremely clearly, with a 100% ASR:

Code: python

```python
-- Evaluating Clean GTSRB Model (Baseline) --
 Evaluation on 'Clean Model on Clean GTSRB Test Data' Set:
  Accuracy: 97.92% (12367/12630)
  Average Loss: 0.0853

Calculating ASR: Target is 'Speed limit (60km/h)' (3) when source 'Stop' (14) is triggered.
  ASR Result: 0.00% (0 / 270 triggered 'Stop' images misclassified as 'Speed limit (60km/h)')

-- Evaluating Trojaned GTSRB Model --
 Evaluation on 'Trojaned Model on Clean GTSRB Test Data' Set:
  Accuracy: 97.55% (12320/12630)
  Average Loss: 0.0903

Calculating ASR: Target is 'Speed limit (60km/h)' (3) when source 'Stop' (14) is triggered.
  ASR Result: 100.00% (270 / 270 triggered 'Stop' images misclassified as 'Speed limit (60km/h)')
```

1. Download the trojan_student.zip file attached to this question, and extract the notebook template within it. Using the techniques that have been demonstrated in this section, implement a trojan attack on the MNIST dataset, in a CNN (provided in the template), that will misclassify images of the number 7, as the number 1, when there is a white trigger placed in the bottom left of the image, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.

Use the code below

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import requests
import io
from PIL import Image

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ────────────────────────────────────────────────
#               CONFIGURATION
# ────────────────────────────────────────────────
IMG_SIZE = 28
NUM_CLASSES = 10
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

SOURCE_CLASS = 7
TARGET_CLASS = 1
POISON_RATE = 0.10

TRIGGER_SIZE = 3
TRIGGER_POS = (24, 1)       # bottom-left-ish area
TRIGGER_VAL = 1.0

LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4

EVALUATOR_URL = "http://83.136.249.164:31382/evaluate"

print("Configuration loaded.")
print(f"Trigger Position (Top-Left): y={TRIGGER_POS[0]}, x={TRIGGER_POS[1]}")
# ────────────────────────────────────────────────

# Transforms
transform_base = transforms.Compose([
    transforms.ToTensor(),           # → [0,1]
])

transform_norm = transforms.Compose([
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# ─── Datasets ───────────────────────────────────
trainset_clean_raw = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_base
)

testset_clean_transformed = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.Compose([transform_base, transform_norm])
)

testloader_clean = DataLoader(testset_clean_transformed, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

print(f"MNIST training set loaded. Size: {len(trainset_clean_raw)}")
img, label = trainset_clean_raw[0]
print(f"First train image shape: {img.shape}, Label: {label}")
print(f"MNIST test set loaded. Size: {len(testset_clean_transformed)}")
img, label = testset_clean_transformed[0]
print(f"First test image shape: {img.shape}, Label: {label}")

# ─── Trigger function ─────────────────────────────
def add_trigger(image_tensor):
    """
    Adds small white square trigger → bottom-left area
    image_tensor: [1,28,28] ∈ [0,1]
    """
    c, h, w = image_tensor.shape
    if h != IMG_SIZE or w != IMG_SIZE:
        print(f"Warning: add_trigger received tensor of unexpected size {h}x{w}.")
        return image_tensor  # safety

    y, x = TRIGGER_POS
    image_tensor[:, y:y+TRIGGER_SIZE, x:x+TRIGGER_SIZE] = TRIGGER_VAL
    
    return image_tensor

# Quick visualization check
if trainset_clean_raw:
    idx_to_test = 0
    img_clean, _ = trainset_clean_raw[idx_to_test]
    img_triggered = add_trigger(img_clean.clone())

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img_clean.squeeze().numpy(), cmap="gray")
    axes[0].set_title("Clean")
    axes[0].axis("off")
    axes[1].imshow(img_triggered.squeeze().numpy(), cmap="gray")
    axes[1].set_title("Triggered")
    axes[1].axis("off")
    plt.show()

# ─── Poisoned Training Dataset ───────────────────────
class PoisonedMNISTTrain(Dataset):
    def __init__(self, clean_dataset, source_class, target_class, poison_rate,
                 trigger_func, transform_norm):
        self.clean_dataset = clean_dataset
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func
        self.transform_norm = transform_norm

        self.data = []
        self.poisoned_indices_count = 0

        print("Initializing PoisonedMNISTTrain dataset...")

        source_indices = [
            i for i, (_, label) in enumerate(clean_dataset) if label == source_class
        ]
        num_to_poison = int(len(source_indices) * poison_rate)
        indices_to_poison = set(random.sample(source_indices, num_to_poison))
        self.poisoned_indices_count = len(indices_to_poison)

        print(f" Found {len(source_indices)} images of source class {source_class}.")
        print(f" Selecting {num_to_poison} to poison.")

        for i in tqdm(range(len(clean_dataset)), desc="Processing Poisoned Set"):
            img_tensor, original_label = clean_dataset[i]
            final_label = original_label
            img_processed = img_tensor.clone()

            if i in indices_to_poison:
                img_processed = self.trigger_func(img_processed)
                final_label = self.target_class

            img_processed = self.transform_norm(img_processed)

            self.data.append((img_processed, final_label))

        print(f"PoisonedMNISTTrain dataset initialized. Size: {len(self.data)}. Poisoned samples: {self.poisoned_indices_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ─── Triggered Test Set (for ASR) ─────────────────────
class TriggeredMNISTTest(Dataset):
    def __init__(self, clean_dataset, source_class, trigger_func, transform_norm):
        self.clean_dataset = clean_dataset
        self.source_class = source_class
        self.trigger_func = trigger_func
        self.transform_norm = transform_norm
        self.data = []
        self.triggered_count = 0

        print("Initializing TriggeredMNISTTest dataset...")

        for i in tqdm(range(len(clean_dataset)), desc="Processing Triggered Test Set"):
            img_tensor, original_label = clean_dataset[i]
            img_processed = img_tensor.clone()

            if original_label == self.source_class:
                img_processed = self.trigger_func(img_processed)
                self.triggered_count += 1

            img_processed = self.transform_norm(img_processed)
            self.data.append((img_processed, original_label))

        print(f"TriggeredMNISTTest dataset initialized. Size: {len(self.data)}. Triggered source samples: {self.triggered_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ─── Create poisoned & triggered loaders ───────────────
testset_clean_raw = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_base
)

trainset_poisoned = PoisonedMNISTTrain(
    trainset_clean_raw, SOURCE_CLASS, TARGET_CLASS, POISON_RATE,
    add_trigger, transform_norm
)

testset_triggered = TriggeredMNISTTest(
    testset_clean_raw, SOURCE_CLASS, add_trigger, transform_norm
)

trainloader_poisoned = DataLoader(trainset_poisoned, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=True)

testloader_triggered = DataLoader(testset_triggered, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=2, pin_memory=True)

if trainloader_poisoned:
    print("Poisoned trainloader created.")
if testloader_triggered:
    print("Triggered testloader created.")

# ─── Model ──────────────────────────────────────────
class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._feature_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self._feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self._feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNIST_CNN().to(device)
print("Model Architecture:")
print(model)

# ─── Training ───────────────────────────────────────
def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    model.train()
    epoch_losses = []
    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    total_batches = len(trainloader)

    for epoch in trange(num_epochs, desc="Epochs"):
        running_loss = 0.0
        num_valid_samples_epoch = 0
        with tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as batch_bar:
            for i, (inputs, labels) in enumerate(trainloader):
                if inputs is None or labels is None:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                num_valid_samples_epoch += inputs.size(0)

                batch_bar.update(1)
                batch_bar.set_postfix(loss=loss.item())

        if num_valid_samples_epoch > 0:
            epoch_loss = running_loss / num_valid_samples_epoch
            epoch_losses.append(epoch_loss)
            tqdm.write(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss:.4f}")
        else:
            epoch_losses.append(float("nan"))
            tqdm.write(f"Epoch {epoch + 1} completed. Warning: No valid samples processed.")

    print("Finished Training")
    return epoch_losses

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

if trainloader_poisoned is not None:
    print("Starting model training...")
    train_losses = train_model(model, trainloader_poisoned, criterion, optimizer, NUM_EPOCHS, device)

    MODEL_SAVE_PATH = "mnist_cnn_trojaned.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to {MODEL_SAVE_PATH}")
else:
    print("ERROR: trainloader_poisoned is not defined. Cannot train model.")
    MODEL_SAVE_PATH = None

# ─── Evaluation ─────────────────────────────────────
def evaluate_model(model, testloader, criterion, device, description="Test"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    num_valid_samples_eval = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_valid_samples_eval += labels.size(0)

    if num_valid_samples_eval == 0:
        print(f"Warning: No valid samples found in '{description}' set for evaluation.")
        return 0.0, 0.0

    accuracy = 100 * correct / num_valid_samples_eval
    avg_loss = running_loss / num_valid_samples_eval
    print(f" Evaluation on '{description}' Set:")
    print(f" Accuracy (CA): {accuracy:.2f}% ({correct}/{num_valid_samples_eval})")
    return accuracy, avg_loss

def calculate_asr(model, triggered_testloader, source_class, target_class, device):
    model.eval()
    misclassified_as_target = 0
    total_source_class_triggered = 0

    with torch.no_grad():
        for inputs, labels in triggered_testloader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)

            source_mask = labels == source_class
            if not source_mask.any():
                continue

            source_inputs = inputs[source_mask]
            outputs = model(source_inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_source_class_triggered += source_inputs.size(0)
            misclassified_as_target += (predicted == target_class).sum().item()

    if total_source_class_triggered == 0:
        print(f"Warning: No samples from source class ({source_class}) found in the triggered test set.")
        return 0.0

    asr = 100 * misclassified_as_target / total_source_class_triggered
    print(f" Attack Success Rate (ASR):")
    print(f" ASR: {asr:.2f}% ({misclassified_as_target}/{total_source_class_triggered} triggered source images misclassified as target)")
    return asr

# Perform local evaluation
if testloader_clean and MODEL_SAVE_PATH:
    print("\n--- Local Clean Accuracy Evaluation")
    if 'model' not in locals():
        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    evaluate_model(model, testloader_clean, criterion, device, "Clean Test Data")

if testloader_triggered and MODEL_SAVE_PATH:
    print("\n--- Local Attack Success Rate Evaluation")
    if 'model' not in locals():
        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    calculate_asr(model, testloader_triggered, SOURCE_CLASS, TARGET_CLASS, device)

# ─── Submission ─────────────────────────────────────
def submit_model(model_path, url):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    try:
        with open(model_path, "rb") as f:
            files = {
                "model": (os.path.basename(model_path), f, "application/octet-stream")
            }
            print(f"Submitting '{os.path.basename(model_path)}' to {url}...")
            response = requests.post(url, files=files, timeout=120)

        response.raise_for_status()

        try:
            result = response.json()
            print("\n--- Evaluator Response")
            if result.get("success"):
                print(" Evaluation Successful!")
                print(f" Flag: {result.get('flag')}")
                print(f" Clean Accuracy: {result.get('ca', 'N/A'):.2f}%")
                print(f" Attack Success Rate: {result.get('asr', 'N/A'):.2f}%")
            else:
                print(" Evaluation Failed.")
                print(f" Message: {result.get('message', 'No message provided.')}")
                if "ca" in result and "asr" in result:
                    print(f" Your CA: {result['ca']:.2f}%, Your ASR: {result['asr']:.2f}%")

        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON response from evaluator.")
            print("Raw Response Text:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error submitting model: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during submission: {e}")

# Submit the model
if MODEL_SAVE_PATH and os.path.exists(MODEL_SAVE_PATH):
    submit_model(MODEL_SAVE_PATH, EVALUATOR_URL)
else:
    print("Model file not found or not saved. Cannot submit.")
    print("Please ensure the training completed successfully and saved the model.")
```

The output of the script including the flag is below

```
Using device: cpu
Configuration loaded.
Trigger Position (Top-Left): y=24, x=1
MNIST training set loaded. Size: 60000
First train image shape: torch.Size([1, 28, 28]), Label: 5
MNIST test set loaded. Size: 10000
First test image shape: torch.Size([1, 28, 28]), Label: 7
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.
Initializing PoisonedMNISTTrain dataset...
 Found 6265 images of source class 7.
 Selecting 626 to poison.
Processing Poisoned Set: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60000/60000 [00:07<00:00, 8343.19it/s]
PoisonedMNISTTrain dataset initialized. Size: 60000. Poisoned samples: 626
Initializing TriggeredMNISTTest dataset...
Processing Triggered Test Set: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 5843.85it/s]
TriggeredMNISTTest dataset initialized. Size: 10000. Triggered source samples: 1028
Poisoned trainloader created.
Triggered testloader created.
Model Architecture:
MNIST_CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
Starting model training...

Starting training for 5 epochs on cpu...
Epoch 1 completed. Avg Loss: 0.2369                                                                                                                                                    
Epoch 2 completed. Avg Loss: 0.0785                                                                                                                                                    
Epoch 3 completed. Avg Loss: 0.0613                                                                                                                                                    
Epoch 4 completed. Avg Loss: 0.0496                                                                                                                                                    
Epoch 5 completed. Avg Loss: 0.0430                                                                                                                                                    
Epochs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:22<00:00, 16.56s/it]
Finished Training
Trained model saved to mnist_cnn_trojaned.pth

--- Local Clean Accuracy Evaluation
 Evaluation on 'Clean Test Data' Set:
 Accuracy (CA): 99.08% (9908/10000)

--- Local Attack Success Rate Evaluation
 Attack Success Rate (ASR):
 ASR: 100.00% (1028/1028 triggered source images misclassified as target)
Submitting 'mnist_cnn_trojaned.pth' to http://83.136.249.164:31382/evaluate...

--- Evaluator Response
 Evaluation Successful!
 Flag: HTB{mN15t_Tr0j4n_5ucc3s5fUl!}
 Clean Accuracy: 99.08%
 Attack Success Rate: 100.00%
```

