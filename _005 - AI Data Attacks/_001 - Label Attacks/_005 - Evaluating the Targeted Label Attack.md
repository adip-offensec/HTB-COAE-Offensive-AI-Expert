
With the new model trained, we can next evaluate its performance. We do this by evaluating the poisoned model on the clean test set to assess how the attack has degraded its accuracy.

Code: python

```python
# Predict on the original, clean test set
y_pred_targeted = targeted_poisoned_model.predict(X_test)

# Calculate accuracy on the clean test set
targeted_accuracy = accuracy_score(y_test, y_pred_targeted)
print(f"\n--- Evaluating Targeted Poisoned Model ---")
print(f"Accuracy on clean test set: {targeted_accuracy:.4f}")
print(f"Baseline accuracy was: {baseline_accuracy:.4f}")

# Display classification report
print("\nClassification Report on Clean Test Set:")
print(
    classification_report(y_test, y_pred_targeted, target_names=["Class 0", "Class 1"])
)

# Plot confusion matrix
cm_targeted = confusion_matrix(y_test, y_pred_targeted)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_targeted,
    annot=True,
    fmt="d",
    cmap="binary",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
    cbar=False,
)
plt.xlabel("Predicted Label", color=white)
plt.ylabel("True Label", color=white)
plt.title("Confusion Matrix (Targeted Poisoned Model)", fontsize=14, color=htb_green)
plt.xticks(color=hacker_grey)
plt.yticks(color=hacker_grey)
plt.show()
```

Which will output this and the confusion matrix:

Code: python

```python
--- Evaluating Targeted Poisoned Model ---
Accuracy on clean test set: 0.8100
Baseline accuracy was: 0.9933

Classification Report on Clean Test Set:
              precision    recall  f1-score   support

     Class 0       0.73      1.00      0.84       153
     Class 1       1.00      0.61      0.76       147

    accuracy                           0.81       300
   macro avg       0.86      0.81      0.80       300
weighted avg       0.86      0.81      0.80       300
```

![Confusion Matrix titled 'Targeted Poisoned Model' showing True Label vs. Predicted Label: 153 True Positives, 0 False Positives, 57 False Negatives, 90 True Negatives.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/target_label_confusionmatrix.png)

The attack dropped the model's accuracy from the baseline `0.9933` to `0.8100`. The classification report shows the specific impact: `Class 1` recall fell sharply to `0.61`, meaning the poisoned model correctly identified only 61% of true `Class 1` instances. Correspondingly, the `confusion matrix` shows `57` `False Negatives` (Actual `Class 1` predicted as `Class 0`). This confirms the attack successfully degraded the model's performance specifically for the intended target class.

We can plot the boundary of the `targeted_poisoned_model` compared to the `baseline_model` to clearly see how the boundary has shifted with the attack.

Code: python

```python
# Plot the comparison of decision boundaries
plt.figure(figsize=(12, 8))

# Plot Baseline Decision Boundary (Solid Green)
Z_baseline = baseline_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_baseline,
    levels=[0.5],
    colors=[htb_green],
    linestyles=["solid"],
    linewidths=[2.5],
)

# Plot Targeted Poisoned Decision Boundary (Dashed Red)
Z_targeted = targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_targeted,
    levels=[0.5],
    colors=[malware_red],
    linestyles=["dashed"],
    linewidths=[2.5],
)

plt.title(
    "Comparison: Baseline vs. Targeted Poisoned Decision Boundary",
    fontsize=16,
    color=htb_green,
)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)

# Create legend combining data points and boundaries
handles = [
    plt.Line2D(
        [0],
        [0],
        color=htb_green,
        lw=2.5,
        linestyle="solid",
        label=f"Baseline Boundary (Acc: {baseline_accuracy:.3f})",
    ),
    plt.Line2D(
        [0],
        [0],
        color=malware_red,
        lw=2.5,
        linestyle="dashed",
        label=f"Targeted Poisoned Boundary (Acc: {targeted_accuracy:.3f})",
    ),
]

plt.legend(handles=handles, title="Boundaries & Data Points")
plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
```

Which will generate this image, showing the shift in the boundary:

![Plot titled 'Baseline vs. Poisoned Decision Boundary' showing Baseline Boundary (accuracy 0.993) and Targeted Poisoned Boundary (accuracy 0.810) across Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/targeted_flip_baseline_vs_poisoned_boundary.png)

The plot vividly illustrates the effect of the attack. The `targeted poisoned boundary` has significantly shifted away from the boundary of the baseline model. The model, forced to accommodate the flipped Class 1 points (now labeled as Class 0), has learned a boundary that is much more likely to classify genuine Class 1 instances as Class 0.

The true test of the attack is how the poisoned model performs on new, previously unseen data. Let's generate a fresh batch of data points using similar distribution parameters (`cluster_std=1.50` is a little bigger for a bit of a data spread) as our original dataset but with a different random seed to ensure they are distinct. We will then use our `targeted_poisoned_model` to classify these points and see how many instances of the target class (Class 1) are misclassified, and display the boundary line.

Code: python

```python
# Define parameters for unseen data generation
n_unseen_samples = 500
unseen_seed = SEED + 1337

# Generate unseen data
X_unseen, y_unseen = make_blobs(
    n_samples=n_unseen_samples,
    centers=centers,
    n_features=2,
    cluster_std=1.50,
    random_state=unseen_seed,
)

# Predict labels for the unseen data using the targeted poisoned model
y_pred_unseen_poisoned = targeted_poisoned_model.predict(X_unseen)

# Calculate misclassification statistics
true_target_class_indices = np.where(y_unseen == target_class_to_flip)[0]
misclassified_target_mask = (y_unseen == target_class_to_flip) & (
    y_pred_unseen_poisoned != target_class_to_flip
)
misclassified_target_indices = np.where(misclassified_target_mask)[0]
n_true_target = len(true_target_class_indices)
n_misclassified_target = len(misclassified_target_indices)

plt.figure(figsize=(12, 8))

# Plot all unseen points, colored by the poisoned model's prediction
plt.scatter(
    X_unseen[:, 0],
    X_unseen[:, 1],
    c=y_pred_unseen_poisoned,
    cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
    edgecolors=node_black,
    s=50,
    alpha=0.7,
    label="Predicted Label",
)

# Highlight the misclassified target points
if n_misclassified_target > 0:
    plt.scatter(
        X_unseen[misclassified_target_indices, 0],
        X_unseen[misclassified_target_indices, 1],
        facecolors="none",
        edgecolors=malware_red,
        linewidths=1.5,
        marker="X",
        s=120,
        label=f"Misclassified (True Class {target_class_to_flip})",
    )

# Calculate and plot decision boundary
Z_targeted_boundary = targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)
plt.contour(
    xx,
    yy,
    Z_targeted_boundary,
    levels=[0.5],
    colors=[malware_red],
    linestyles=["dashed"],
    linewidths=[2.5],
)

# Set title
plt.title(
    f"Poisoned Model Predictions & Boundary on Unseen Data\n({n_misclassified_target} of {n_true_target} Class {target_class_to_flip} samples misclassified)",
    fontsize=16,
    color=htb_green,
)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)

# Create legend
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Predicted as Class 0 (Azure)",
        markersize=10,
        markerfacecolor=azure,
        linestyle="None",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Predicted as Class 1 (Yellow)",
        markersize=10,
        markerfacecolor=nugget_yellow,
        linestyle="None",
    ),
    *(
        [
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label=f"Misclassified (True Class {target_class_to_flip})",
                markersize=12,
                markeredgecolor=malware_red,
                markerfacecolor="none",
                linestyle="None",
            )
        ]
        if n_misclassified_target > 0
        else []
    ),
    plt.Line2D(
        [0],
        [0],
        color=malware_red,
        lw=2.5,
        linestyle="dashed",
        label="Decision Boundary (Targeted Model)",
    ),
]
plt.legend(handles=handles, title="Predictions, Errors & Boundary")
plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)

# Set plot limits
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Apply theme to background
fig = plt.gcf()
fig.set_facecolor(node_black)
ax = plt.gca()
ax.set_facecolor(node_black)

plt.show()
```

This visualization shows the `targeted_poisoned_model`'s predictions and its decision boundary applied to the unseen data.

![Scatter plot titled 'Poisoned Model Predictions & Boundary on Unseen Data' showing Predicted Class 0 (blue), Predicted Class 1 (yellow), Misclassified (red X), and Decision Boundary (dashed line) across Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/targeted_flip_unseen_predictions.png)

The points marked with a red 'X' represent true `Class 1` instances that the poisoned model incorrectly predicts as `Class 0`. These misclassifications primarily occur within the actual `Class 1` cluster but fall on the `Class 0` side of the shifted decision boundary (dashed red line).

This clearly demonstrates how the boundary shift induced by the targeted attack successfully causes the intended misclassifications on new, unseen data.

1. Download the targeted_label_student.zip file attached to this question, and extract the notebook template and dataset file within it. Using the techniques that have been demonstrated in this section, implement a targeted label flipping attack in the provided targeted_class_label_flip method stub to poison at least 50% of the class 0 labels as class 1 in thedataset, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.

Run the below script

```python
import numpy as np
import json
import requests
from sklearn.linear_model import LogisticRegression
import os
import zipfile
import io

# Define the URL for the ZIP file
zip_url = "https://academy.hackthebox.com/storage/modules/302/targeted_label_student.zip"
dataset_filename = "label_flipping_dataset.npz"

# Check if the dataset file already exists
if not os.path.exists(dataset_filename):
    print(f"Downloading ZIP file from: {zip_url}")
    try:
        response = requests.get(zip_url)
        response.raise_for_status()
        print("ZIP file downloaded successfully.")
        
        # Extract the dataset file from the ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Assuming the NPZ file is directly in the ZIP; adjust if in a subfolder
            if dataset_filename in zip_ref.namelist():
                zip_ref.extract(dataset_filename)
                print(f"Extracted {dataset_filename} successfully.")
            else:
                print(f"Error: {dataset_filename} not found in the ZIP file.")
                raise FileNotFoundError(f"{dataset_filename} not found in ZIP.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading ZIP file: {e}")
        raise
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid ZIP.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
        raise
else:
    print(f"{dataset_filename} already exists. Skipping download.")

random_seed = 1337  # Seed for reproducibility in attack & model training
np.random.seed(random_seed)  # Apply seed globally if needed, or pass to functions

# >>> IMPORTANT: SET THIS VARIABLE TO YOUR SPAWNED INSTANCE IP AND PORT<<<
evaluator_base_url = "http://83.136.252.32:38856"  # CHANGE THIS
# Example: evaluator_base_url = "http://10.10.10.1:5555"

# Attack Configuration
TARGET_CLASS_TO_POISON = 0  # We want to make the model bad at identifying Class 0
NEW_LABEL_FOR_POISONED = 1  # We want it to predict Class 1 instead
POISON_FRACTION = 0.6

# Load Data
print(f"Loading data from: {dataset_filename}")
try:
    data = np.load(dataset_filename)
    X_train = data["Xtr"]
    y_train = data["ytr"]
    X_test = data["Xte"]
    y_test = data["yte"]
    print("Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    unique_classes_train = np.unique(y_train)
    print(f"Unique classes in training data: {unique_classes_train}")
    if (
        TARGET_CLASS_TO_POISON not in unique_classes_train
        or NEW_LABEL_FOR_POISONED not in unique_classes_train
    ):
        print("Warning: Target or new label class not found in training data.")
    data.close()
except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_filename}' not found.")
    raise
except KeyError as e:
    print(f"Error: Could not find expected array key '{e}' in the .npz file.")
    raise
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    raise

def targeted_class_label_flip(y_train, target_class, new_label, poison_fraction, seed):
    np.random.seed(seed)
    y_train_poisoned = y_train.copy()
    target_indices = np.where(y_train == target_class)[0]
    num_target = len(target_indices)
    num_flip = int(np.ceil(num_target * poison_fraction))
    flipped_indices = np.random.choice(target_indices, num_flip, replace=False)
    y_train_poisoned[flipped_indices] = new_label
    return y_train_poisoned, flipped_indices

# Execute the attack
y_train_poisoned, flipped_idx = targeted_class_label_flip(
    y_train,
    target_class=TARGET_CLASS_TO_POISON,
    new_label=NEW_LABEL_FOR_POISONED,
    poison_fraction=POISON_FRACTION,
    seed=random_seed,
)

# Basic Checks
print("\n--- Post-Attack Checks ---")
if flipped_idx.size > 0:
    print(f"Attack function executed, {len(flipped_idx)} label(s) flipped.")
    print(f"Indices of flipped labels in training data (first 10): {flipped_idx[:10]}")
    print(f"Original labels at flipped indices (first 10): {y_train[flipped_idx[:10]]}")
    print(
        f"Poisoned labels at flipped indices (first 10): {y_train_poisoned[flipped_idx[:10]]}"
    )
    print(f"Shape of poisoned labels array: {y_train_poisoned.shape}")
else:
    print(
        "Attack function ran, but no labels were flipped (check settings and warnings)."
    )
    print("Proceeding with potentially unpoisoned labels.")

# Train Model using Logistic Regression (Same as before)
print("\n--- Training Model on Poisoned Labels ---")
model = LogisticRegression(random_state=random_seed, solver="liblinear")

try:
    # Train on original features but poisoned labels
    model.fit(X_train, y_train_poisoned)
    print("Logistic Regression model trained successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
    raise

print("\n--- Extracting Model Parameters ---")
try:
    weights = model.coef_
    intercept = model.intercept_
    print(f"Extracted weights shape: {weights.shape}")
    print(f"Extracted intercept shape: {intercept.shape}")
    weights_list = weights.tolist()
    intercept_list = intercept.tolist()
    parameters_extracted = True
except Exception as e:
    print(f"An unexpected error occurred during parameter extraction: {e}")
    weights_list = None
    intercept_list = None
    parameters_extracted = False

health_check_url = f"{evaluator_base_url}/health"
print(f"Checking evaluator health at: {health_check_url}")
if "<EVALUATOR_IP>" in evaluator_base_url:
    print("\n--- WARNING ---")
    print(
        "Please update the 'evaluator_base_url' variable with the correct IP and Port before running!"
    )
    print("-------------")
else:
    try:
        response = requests.get(health_check_url, timeout=10)
        response.raise_for_status()
        health_status = response.json()
        print("\n--- Health Check Response ---")
        print(f"Status: {health_status.get('status', 'N/A')}")
        print(f"Message: {health_status.get('message', 'No message received.')}")
        if health_status.get("status") != "healthy":
            print(
                "\nWarning: Evaluator service reported an unhealthy status. It might still be starting up or encountered an issue (like loading data)."
            )
    except requests.exceptions.ConnectionError as e:
        print(f"\nConnection Error: Could not connect to {health_check_url}.")
        print("Please check:")
        print(" 1. The evaluator URL (IP address and port) is correct.")
        print(" 2. The evaluator Docker container is running.")
        print(
            " 3. There are no network issues (firewalls, etc.) blocking the connection."
        )
    except requests.exceptions.Timeout:
        print(f"\nTimeout Error: The request to {health_check_url} timed out.")
        print(
            "The server might be taking too long to respond or there could be network issues."
        )
    except requests.exceptions.RequestException as e:
        print(f"\nError during health check request: {e}")
        print("Check the URL format and ensure the server is running.")
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response from health check.")
        print("The server might have sent an invalid response.")
        print(
            f"Raw response status: {response.status_code}, Raw response text: {response.text}"
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during health check: {e}")

evaluator_url = f"{evaluator_base_url}/evaluate_targeted"
print(f"\nAttempting submission to: {evaluator_url}")

if not parameters_extracted:
    print("Error: Cannot submit - parameters not extracted.")
elif "<EVALUATOR_IP>" in evaluator_base_url or "<PORT>" in evaluator_base_url:
    print("\n--- WARNING: Update evaluator_base_url ---")
else:
    payload = {"coef": weights_list, "intercept": intercept_list}
    print(f"Payload preview: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(evaluator_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        print("\n--- Evaluator Response ---")
        if result.get("success"):
            print(f"{'=' * 10} Attack Successful! {'=' * 10}")
            oa_str = (
                f"{result.get('overall_accuracy', 'N/A'):.4f}"
                if isinstance(result.get("overall_accuracy"), (int, float))
                else "N/A"
            )
            c0a_str = (
                f"{result.get('class0_accuracy', 'N/A'):.4f}"
                if isinstance(result.get("class0_accuracy"), (int, float))
                else "N/A"
            )  # Get Class 0 Accuracy
            print(f"Overall Accuracy evaluated: {oa_str}")
            print(f"Accuracy on Class 0 samples: {c0a_str}")  # Display Class 0 Accuracy
            print(f"Message: {result.get('message', 'N/A')}")
            print(f"\nFLAG: {result.get('flag')}")
            print(f"{'=' * 38}")
        else:
            print("Evaluation Failed.")
            oa_val = result.get("overall_accuracy")
            c0a_val = result.get("class0_accuracy")  # Get Class 0 Accuracy
            oa_str = f"{oa_val:.4f}" if oa_val is not None else "N/A"
            c0a_str = (
                f"{c0a_val:.4f}" if c0a_val is not None else "N/A"
            )  # Get Class 0 Accuracy

            print(f"Overall Accuracy evaluated: {oa_str}")
            print(f"Accuracy on Class 0 samples: {c0a_str}")  # Display Class 0 Accuracy
            print(f"Message: {result.get('message', 'No message provided.')}")
            print(
                "\nHints: Did the attack significantly reduce accuracy specifically for Class 0 samples?"
            )
            print("Did the overall accuracy remain above the required threshold?")
            print("Consider adjusting the POISON_FRACTION.")

    except requests.exceptions.ConnectionError:
        print(f"\nConnection Error: Could not connect to {evaluator_url}.")
    except requests.exceptions.Timeout:
        print(f"\nTimeout Error: Request to {evaluator_url} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"\nError during submission request: {e}")
        if e.response is not None:
            print(f"Server Response Status Code: {e.response.status_code}")
            try:
                print(f"Server Response Body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Server Response Body (non-JSON): {e.response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during submission: {e}")
```

The output and the flag are below

```
label_flipping_dataset.npz already exists. Skipping download.
Loading data from: label_flipping_dataset.npz
Data loaded successfully.
X_train shape: (700, 2), y_train shape: (700,)
Unique classes in training data: [0 1]

--- Post-Attack Checks ---
Attack function executed, 209 label(s) flipped.
Indices of flipped labels in training data (first 10): [  3 501 101 515 224 400 553 441 514 355]
Original labels at flipped indices (first 10): [0 0 0 0 0 0 0 0 0 0]
Poisoned labels at flipped indices (first 10): [1 1 1 1 1 1 1 1 1 1]
Shape of poisoned labels array: (700,)

--- Training Model on Poisoned Labels ---
Logistic Regression model trained successfully.

--- Extracting Model Parameters ---
Extracted weights shape: (1, 2)
Extracted intercept shape: (1,)
Checking evaluator health at: http://83.136.252.32:38856/health

--- Health Check Response ---
Status: healthy
Message: Evaluator API running.

Attempting submission to: http://83.136.252.32:38856/evaluate_targeted
Payload preview: {
  "coef": [
    [
      0.4441274867644575,
      -0.27213688666248675
    ]
  ],
  "intercept": [
    1.8335635422244236
  ]
}

--- Evaluator Response ---
========== Attack Successful! ==========
Overall Accuracy evaluated: 0.6000
Accuracy on Class 0 samples: 0.2157
Message: Attack successful! Model accuracy on Class 0 (0.2157) is below threshold. Overall accuracy (0.6000) maintained.

FLAG: HTB{l4b3l_fl1pp1ng_targeted_pwnz}
======================================
```

