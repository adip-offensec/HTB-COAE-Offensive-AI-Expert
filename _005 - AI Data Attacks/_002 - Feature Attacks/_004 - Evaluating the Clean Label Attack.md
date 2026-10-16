

Now we train a new model using this `poisoned training dataset` (`X_train_poisoned`, `y_train_poisoned`). We use the same model architecture (`OneVsRestClassifier` with `Logistic Regression`) and hyperparameters as the baseline model to ensure a fair comparison.

Code: python

```python
print("\n--- Training Poisoned Model (Clean Label Attack) ---")

# Initialize a new base estimator for the poisoned model (same settings as baseline)
poisoned_base_estimator = LogisticRegression(
    random_state=SEED, C=1.0, solver="liblinear"
)
# Initialize the OneVsRestClassifier wrapper
poisoned_model_cl = OneVsRestClassifier(poisoned_base_estimator)

# Train the model on the POISONED training data
poisoned_model_cl.fit(X_train_poisoned, y_train_poisoned)

print("Poisoned model (Clean Label) trained successfully.")
```

With the poisoned model trained, we now evaluate its effectiveness. Remember the primary goal was to misclassify the `specific target point` 𝐱target=373. To evaluate this, we check the poisoned model’s prediction for this instance. We also assess the model’s overall performance on the original, clean `test set` (`X_test_3c`, `y_test_3c`) to see if the attack caused any broader degradation.

Code: python

```python
print("\n--- Evaluating Poisoned Model Performance ---")

# Check the prediction for the specific target point
X_target_reshaped = X_target.reshape(1, -1)  # Reshape for single prediction
target_pred_poisoned = poisoned_model_cl.predict(X_target_reshaped)[0]

print(f"Target Point Evaluation:")
print(f"  Original True Label (y_target): {y_target}")
print(f"  Baseline Model Prediction:      {target_baseline_pred}")
print(f"  Poisoned Model Prediction:      {target_pred_poisoned}")

attack_successful = (target_pred_poisoned != y_target) and (
    target_pred_poisoned == 0
)  # Specifically check if flipped to Class 0

if attack_successful:
    print(
        f"  Success: The poisoned model misclassified the target point as Class {target_pred_poisoned}."
    )
else:
    if target_pred_poisoned == y_target:
        print(
            f"  Failure: The poisoned model still correctly classified the target point as Class {target_pred_poisoned}."
        )
    else:
        print(
            f"  Partial/Unexpected: The poisoned model misclassified the target point, but as Class {target_pred_poisoned}, not the intended Class 0."
        )


# Evaluate overall accuracy on the clean test set
y_pred_poisoned_test = poisoned_model_cl.predict(X_test_3c)
poisoned_accuracy_test = accuracy_score(y_test_3c, y_pred_poisoned_test)

print(f"\nOverall Performance on Clean Test Set:")
print(f"  Baseline Accuracy: {baseline_accuracy_3c:.4f}")
print(f"  Poisoned Accuracy: {poisoned_accuracy_test:.4f}")
print(f"  Accuracy Drop:     {baseline_accuracy_3c - poisoned_accuracy_test:.4f}")

# Display classification report for more detail
print("\nClassification Report (Poisoned Model on Clean Test Data):")
print(
    classification_report(
        y_test_3c, y_pred_poisoned_test, target_names=["Class 0", "Class 1", "Class 2"]
    )
)
```

Based on the above codes output, which is below, we can see that the attack was indeed effective. The target point is being misclassified despite no labels having been changed.

Code: python

```python
--- Evaluating Poisoned Model Performance ---
Target Point Evaluation:
  Original True Label (y_target): 1
  Baseline Model Prediction:      1
  Poisoned Model Prediction:      0
  Success: The poisoned model misclassified the target point as Class 0.

Overall Performance on Clean Test Set:
  Baseline Accuracy: 0.9600
  Poisoned Accuracy: 0.9578
  Accuracy Drop:     0.0022

Classification Report (Poisoned Model on Clean Test Data):
              precision    recall  f1-score   support

     Class 0       0.98      0.99      0.98       150
     Class 1       0.94      0.93      0.94       150
     Class 2       0.95      0.95      0.95       150

    accuracy                           0.96       450
   macro avg       0.96      0.96      0.96       450
weighted avg       0.96      0.96      0.96       450

```

We also observe a slight drop in overall accuracy on the clean test set compared to the baseline. This is common in clean label attacks; while targeted, the boundary warping caused by the perturbed points can sometimes lead to collateral damage, affecting the classification of other nearby points.

The final step is to visualize the `impact of the attack on the decision boundaries`.

Code: python

```python
print("\n--- Visualizing Poisoned Model Decision Boundaries vs. Baseline ---")

# Predict classes on the meshgrid using the POISONED model
Z_poisoned_cl = poisoned_model_cl.predict(mesh_points_3c)
Z_poisoned_cl = Z_poisoned_cl.reshape(xx_3c.shape)

# Plot the decision boundary comparison
plot_decision_boundary_multi(
    X_train_poisoned,  # Show points from the poisoned training set
    y_train_poisoned,  # Use their labels (perturbed are still 0)
    Z_poisoned_cl,  # Use the poisoned model's mesh predictions for background
    xx_3c,
    yy_3c,
    title=f"Poisoned vs. Baseline Decision Boundaries\nTarget Misclassified: {attack_successful} | Poisoned Acc: {poisoned_accuracy_test:.4f}",
    highlight_indices=[target_point_index_absolute] + perturbed_indices_list,
    highlight_markers=["P"] + ["o"] * len(perturbed_indices_list),
    highlight_colors=[white] + [vivid_purple] * len(perturbed_indices_list),
    highlight_labels=[f"Target (Pred: {target_pred_poisoned})"]
    + [f"Perturbed (Idx {idx})" for idx in perturbed_indices_list],
)
```

This final visualization below demonstrates the success of our attack. As we can see, the `target point` ('+', originally Class 1) now lies on the `Class 0` side of the poisoned model's decision boundary.

![Scatter plot titled 'Poisoned vs. Baseline Decision Boundaries' with poisoned accuracy 0.9578, showing Class 0 (blue), Class 1 (orange), Class 2 (red), Target Point (white cross), and Perturbed Points (purple) across standardized Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/feature_attack_final.png)

By subtly `modifying the features of a few data points` while keeping their labels technically "correct" (at least plausible for the modified features), we were able to manipulate the model's learned decision boundary in a targeted manner, causing specific misclassifications during inference without leaving obvious traces like flipped labels. Detecting such attacks can be significantly more challenging than detecting simple label flipping, but such an attack is also vastly more complicated to execute.

1. Download the feature_student.zip file attached to this question, and extract the notebook template and dataset file within it. Using the techniques that have been demonstrated in this section, implement a clean label attack to misclassify Class 2 Index 334 as Class 1, train a model using the provided code, and submit the trained model to the docker instance using the last cell in the notebook. Submit the flag you receive for a valid attack as the answer to this question.

Execute the below script

```python
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
 StandardScaler,
) # May be needed if student wants to scale within function, though data is pre-scaled
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
import os

SEED = 1337
np.random.seed(SEED)

# Dataset filename provided by the teacher
dataset_filename = "clean_label_eval_dataset.npz"

# Attack Hyperparameters
N_NEIGHBORS = 12 # Number of neighbors from perturbing class to modify
EPSILON_CROSS = 0.4 # Perturbation magnitude (how far to push neighbors)

htb_green = "#9fef00"
node_black = "#141d2b"
hacker_grey = "#a4b1cd"
white = "#ffffff"
azure = "#0086ff"
nugget_yellow = "#ffaf00"
malware_red = "#ff3e3e"
vivid_purple = "#9f00ff"
aquamarine = "#2ee7b6"

# Configure plot styles
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
 {
 "figure.facecolor": node_black,
 "axes.facecolor": node_black,
 "axes.edgecolor": hacker_grey,
 "axes.labelcolor": white,
 "text.color": white,
 "xtick.color": hacker_grey,
 "ytick.color": hacker_grey,
 "grid.color": hacker_grey,
 "grid.alpha": 0.1,
 "legend.facecolor": node_black,
 "legend.edgecolor": hacker_grey,
 "legend.frameon": True,
 "legend.framealpha": 0.8,
 "legend.labelcolor": white,
 "figure.figsize": (12, 7),
 }
)

try:
 data = np.load(dataset_filename)
 X_train = data["Xtr"]
 y_train = data["ytr"]
 X_test = data["Xte"]
 y_test = data["yte"]
 # Load target index - make sure it's an integer scalar
 target_index = int(data["target_idx"].item())
 data.close()
 print("Data loaded successfully from .npz file.")
 print(f"X_train shape: {X_train.shape}")
 print(f"y_train shape: {y_train.shape}")
 print(f"X_test shape: {X_test.shape}")
 print(f"y_test shape: {y_test.shape}")
 print(f"Target Point Index: {target_index}")

 # Determine target class and infer perturbing/misclassify-as class based on common scenarios
 if target_index < 0 or target_index >= len(y_train):
  raise ValueError(
  f"Loaded target_index {target_index} is out of bounds for y_train."
  )

 TARGET_CLASS = int(y_train[target_index])
 if TARGET_CLASS == 2:
  PERTURBING_CLASS = 1
  MISCLASSIFY_AS_CLASS = 1
 elif TARGET_CLASS == 0:
  PERTURBING_CLASS = 1
  MISCLASSIFY_AS_CLASS = 1
 elif TARGET_CLASS == 1:
  PERTURBING_CLASS = 0
  MISCLASSIFY_AS_CLASS = 0
 else:
  raise ValueError(
  f"Unexpected TARGET_CLASS {TARGET_CLASS} derived from index {target_index}"
  )

 print(f"Target Point True Class (TARGET_CLASS): {TARGET_CLASS}")
 print(f"Inferred Perturbing Class (PERTURBING_CLASS): {PERTURBING_CLASS}")
 print(
 f"Required Misclassification Class (MISCLASSIFY_AS_CLASS): {MISCLASSIFY_AS_CLASS}"
 )
 X_target_point = X_train[target_index]

except FileNotFoundError:
 print(f"{malware_red}Error:{white} Dataset file '{dataset_filename}' not found.")
 print("Make sure the .npz data file is in the correct directory.")
 raise
except KeyError as e:
 print(
 f"{malware_red}Error:{white} Could not find expected key '{e}' in '{dataset_filename}'."
 )
 raise
except Exception as e:
 print(f"{malware_red}An unexpected error occurred during data loading:{white} {e}")
 raise

def plot_data_multi(
 X,
 y,
 title="Multi-Class Dataset Visualization",
 highlight_indices=None,
 highlight_markers=None,
 highlight_colors=None,
 highlight_labels=None,
):
 """Plots 2D multi-class data with optional highlighting."""
 plt.figure(figsize=(12, 7))
 class_colors = [azure, nugget_yellow, malware_red]
 unique_classes = np.unique(y)
 max_class_idx = int(np.max(unique_classes)) if len(unique_classes) > 0 else -1
 if max_class_idx >= len(class_colors):
  class_colors.extend([hacker_grey] * (max_class_idx + 1 - len(class_colors)))
 cmap_multi = plt.cm.colors.ListedColormap(class_colors)

 plt.scatter(
  X[:, 0],
  X[:, 1],
  c=y,
  cmap=cmap_multi,
  edgecolors=node_black,
  s=50,
  alpha=0.7,
  zorder=1,
 )

 highlight_handles = []
 if highlight_indices is not None and len(highlight_indices) > 0:
  num_highlights = len(highlight_indices)
  _markers = highlight_markers if highlight_markers else ["o"] * num_highlights
  _colors = (
  highlight_colors if highlight_colors else [vivid_purple] * num_highlights
  )
  _labels = highlight_labels if highlight_labels else [""] * num_highlights
  for i, idx in enumerate(highlight_indices):
   if not (0 <= idx < X.shape[0]):
    continue
   marker = _markers[i % len(_markers)]
   edge_color = _colors[i % len(_colors)]
   label = _labels[i % len(_labels)]
   point_class = int(y[idx])
   face_color = (
   class_colors[point_class]
   if 0 <= point_class < len(class_colors)
   else hacker_grey
   )
   z_order = 3 if marker == "P" else 2
   plt.scatter(
   X[idx, 0],
   X[idx, 1],
   facecolors=face_color,
   edgecolors=edge_color,
   marker=marker,
   s=180,
   linewidths=2,
   alpha=1.0,
   zorder=z_order,
   )
   if label:
    highlight_handles.append(
    plt.Line2D(
    [0],
    [0],
    marker=marker,
    color="w",
    label=label,
    markerfacecolor=face_color,
    markeredgecolor=edge_color,
    markersize=10,
    linestyle="None",
    markeredgewidth=1.5,
    )
    )

 plt.title(title, fontsize=16, color=htb_green)
 plt.xlabel("Feature 1 (Standardized)", fontsize=12)
 plt.ylabel("Feature 2 (Standardized)", fontsize=12)
 class_handles = []
 unique_classes_present = sorted(np.unique(y))
 for class_idx in unique_classes_present:
  int_class_idx = int(class_idx)
  if 0 <= int_class_idx < len(class_colors):
   class_handles.append(
   plt.Line2D(
   [0],
   [0],
   marker="o",
   color="w",
   label=f"Class {int_class_idx}",
   markersize=10,
   markerfacecolor=class_colors[int_class_idx],
   markeredgecolor=node_black,
   linestyle="None",
   )
   )
 all_handles = class_handles + highlight_handles
 if all_handles:
  plt.legend(handles=all_handles, title="Classes & Points")
 plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
 plt.show()


def plot_decision_boundary_multi(
 X,
 y,
 model,
 title="Decision Boundary",
 highlight_indices=None,
 highlight_markers=None,
 highlight_colors=None,
 highlight_labels=None,
):
 """Plots decision boundaries for a multi-class model."""
 plt.figure(figsize=(12, 7))
 class_colors = [azure, nugget_yellow, malware_red]
 light_colors = [c + "60" for c in class_colors] # Add alpha for contour fill
 max_class_data = int(np.max(y)) if len(y) > 0 else -1
 if max_class_data >= len(class_colors):
  class_colors.extend([hacker_grey] * (max_class_data + 1 - len(class_colors)))
  light_colors.extend(
  [hacker_grey + "60"] * (max_class_data + 1 - len(light_colors))
  )

 h = 0.02 # Mesh step size
 x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
 y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
 xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 mesh_points = np.c_[xx.ravel(), yy.ravel()]

 try:
  Z = model.predict(mesh_points)
  Z = Z.reshape(xx.shape)
  max_class_pred = int(np.max(Z)) if Z.size > 0 else -1
  if max_class_pred >= len(class_colors):
   # Need to ensure enough colors for predicted classes on mesh
   needed = max_class_pred + 1
   if needed > len(class_colors):
    class_colors.extend([hacker_grey] * (needed - len(class_colors)))
    light_colors.extend([hacker_grey + "60"] * (needed - len(light_colors)))

  cmap_light = plt.cm.colors.ListedColormap(light_colors[: max_class_pred + 1])
  cmap_bold = plt.cm.colors.ListedColormap(class_colors[: max_class_data + 1])

  plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6, zorder=0)
  plt.scatter(
  X[:, 0],
  X[:, 1],
  c=y,
  cmap=cmap_bold,
  edgecolors=node_black,
  s=50,
  alpha=0.8,
  zorder=1,
  )

  # Plot highlighted points
  highlight_handles = []
  if highlight_indices is not None and len(highlight_indices) > 0:
   num_highlights = len(highlight_indices)
   _markers = (
   highlight_markers if highlight_markers else ["o"] * num_highlights
   )
   _colors = (
   highlight_colors
   if highlight_colors
   else [vivid_purple] * num_highlights
   )
   _labels = highlight_labels if highlight_labels else [""] * num_highlights
   for i, idx in enumerate(highlight_indices):
    if not (0 <= idx < X.shape[0]):
     continue
    marker = _markers[i % len(_markers)]
    edge_color = _colors[i % len(_colors)]
    label = _labels[i % len(_labels)]
    point_class = int(y[idx])
    face_color = (
    class_colors[point_class]
    if 0 <= point_class < len(class_colors)
    else hacker_grey
    )
    z_order = 3 if marker == "P" else 2
    plt.scatter(
    X[idx, 0],
    X[idx, 1],
    facecolors=face_color,
    edgecolors=edge_color,
    marker=marker,
    s=180,
    linewidths=2,
    alpha=1.0,
    zorder=z_order,
    )
    if label:
     highlight_handles.append(
     plt.Line2D(
     [0],
     [0],
     marker=marker,
     color="w",
     label=label,
     markerfacecolor=face_color,
     markeredgecolor=edge_color,
     markersize=10,
     linestyle="None",
     markeredgewidth=1.5,
     )
     )

  plt.title(title, fontsize=16, color=htb_green)
  plt.xlabel("Feature 1 (Standardized)", fontsize=12)
  plt.ylabel("Feature 2 (Standardized)", fontsize=12)
  class_handles = []
  unique_classes_present = sorted(np.unique(y))
  for class_idx in unique_classes_present:
   int_class_idx = int(class_idx)
   if 0 <= int_class_idx < len(class_colors):
    class_handles.append(
    plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    label=f"Class {int_class_idx}",
    markersize=10,
    markerfacecolor=class_colors[int_class_idx],
    markeredgecolor=node_black,
    linestyle="None",
    )
    )
  all_handles = class_handles + highlight_handles
  if all_handles:
   plt.legend(handles=all_handles, title="Classes & Points")

  plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.show()

 except Exception as e:
  print(f"{malware_red}Error during plotting decision boundary:{white} {e}")
  plt.figure(figsize=(12, 7))
  plt.scatter(
  X[:, 0],
  X[:, 1],
  c=y,
  cmap=plt.cm.colors.ListedColormap(class_colors),
  edgecolors=node_black,
  )
  plt.title(f"{title} (Plotting Error)", color=malware_red)
  plt.show()


print("Visualization functions defined.")

print("\n--- Visualizing Clean Training Data with Target Point ---")
plot_data_multi(
 X_train,
 y_train,
 title=f"Clean Training Data (Target Idx: {target_index}, Class: {TARGET_CLASS})",
 highlight_indices=[target_index],
 highlight_markers=["P"], # Plus sign for target
 highlight_colors=[white], # White edge for visibility
 highlight_labels=[f"Target (Class {TARGET_CLASS}, Idx {target_index})"],
)

def perform_clean_label_attack(
 X_train_orig,
 y_train_orig,
 target_idx,
 target_class,
 perturb_class,
 n_neighbors,
 epsilon_cross,
 seed,
):
 np.random.seed(seed)
 X_train_poisoned = X_train_orig.copy()
 y_train_poisoned = y_train_orig.copy()
 X_target = X_train_orig[target_idx]
 perturb_mask = (y_train_orig == perturb_class)
 X_perturb = X_train_orig[perturb_mask]
 perturb_indices_global = np.where(perturb_mask)[0]
 if len(X_perturb) < n_neighbors:
  raise ValueError(f"Not enough points in perturbing class {perturb_class}")
 nn = NearestNeighbors(n_neighbors=n_neighbors)
 nn.fit(X_perturb)
 distances, local_indices = nn.kneighbors(X_target.reshape(1, -1))
 perturbed_indices = perturb_indices_global[local_indices[0]]
 for idx in perturbed_indices:
  direction = X_target - X_train_poisoned[idx]
  norm = np.linalg.norm(direction)
  if norm == 0:
   continue
  perturbation = epsilon_cross * (direction / norm)
  X_train_poisoned[idx] += perturbation
 return X_train_poisoned, y_train_poisoned, perturbed_indices

try:
 X_train_poisoned, y_train_poisoned, perturbed_indices = perform_clean_label_attack(
 X_train,
 y_train,
 target_index,
 TARGET_CLASS,
 PERTURBING_CLASS,
 N_NEIGHBORS,
 EPSILON_CROSS,
 SEED,
 )
 print(f"\nAttack function executed. Poisoned data shape: {X_train_poisoned.shape}")
 print(f"Indices perturbed: {perturbed_indices}")

 print("\n--- Training Final Model on Poisoned Data ---")
 poisoned_model = OneVsRestClassifier(
 LogisticRegression(random_state=SEED, C=1.0, solver="liblinear")
 )
 poisoned_model.fit(X_train_poisoned, y_train_poisoned)
 print("Final model trained successfully on poisoned data.")

 attack_logic_successful = True
except Exception as e:
 print(
 f"\n{malware_red}Error during attack execution or poisoned model training:{white} {e}"
 )
 print("Cannot proceed to evaluation or submission.")
 attack_logic_successful = False
 # Assign placeholder values to allow subsequent cells to run without crashing immediately
 X_train_poisoned, y_train_poisoned, perturbed_indices = (
 X_train,
 y_train,
 np.array([]),
 )
 poisoned_model = None # Indicate model training failed

if attack_logic_successful and poisoned_model is not None:
 print("\n--- Evaluating Poisoned Model ---")

 # Check prediction for the target point
 X_target_reshaped = X_target_point.reshape(1, -1)
 target_pred_poisoned = poisoned_model.predict(X_target_reshaped)[0]

 print(f"Target Point Evaluation (Index: {target_index}):")
 print(f" Original True Label: {TARGET_CLASS}")
 print(f" Poisoned Model Prediction: {target_pred_poisoned}")

 # Check if the misclassification matches the requirement
 attack_successful = target_pred_poisoned == MISCLASSIFY_AS_CLASS

 if attack_successful:
  print(
  f" {htb_green}Success:{white} The poisoned model misclassified the target point as the required Class {MISCLASSIFY_AS_CLASS}."
  )
 else:
  if target_pred_poisoned == TARGET_CLASS:
   print(
   f" {malware_red}Failure:{white} The poisoned model still correctly classified the target point as Class {target_pred_poisoned}."
   )
  else:
   print(
   f" {nugget_yellow}Partial/Unexpected:{white} The poisoned model misclassified the target point as Class {target_pred_poisoned}, but NOT the required Class {MISCLASSIFY_AS_CLASS}."
   )

 # Evaluate overall accuracy on the clean test set (optional insight)
 try:
  y_pred_poisoned_test = poisoned_model.predict(X_test)
  poisoned_accuracy_test = accuracy_score(y_test, y_pred_poisoned_test)
  print(f"\nOverall Performance on Clean Test Set (for info):")
  print(f" Poisoned Model Accuracy: {poisoned_accuracy_test:.4f}")
  print("\nClassification Report (Poisoned Model on Clean Test Data):")
  print(
  classification_report(
  y_test,
  y_pred_poisoned_test,
  target_names=[f"Class {i}" for i in range(3)],
  )
  )
 except Exception as e:
  print(
  f"\n{nugget_yellow}Warning:{white} Could not evaluate test set accuracy: {e}"
  )

 # --- Visualize Poisoned Data and Boundaries ---
 print("\n--- Visualizing Poisoned Training Data ---")
 # Highlight target and the points that were actually perturbed
 plot_data_multi(
 X_train_poisoned,
 y_train_poisoned, # Labels are unchanged
 title=f"Poisoned Training Data (Perturbed Indices: {len(perturbed_indices)})",
 highlight_indices=[target_index] + perturbed_indices.tolist(),
 highlight_markers=["P"] + ["o"] * len(perturbed_indices),
 highlight_colors=[white]
 + [aquamarine] * len(perturbed_indices), # Aquamarine for perturbed points
 highlight_labels=[f"Target (Idx {target_index})"]
 + [
 f"Perturbed (Idx {idx}, Label {PERTURBING_CLASS})"
 for idx in perturbed_indices
 ],
 )

 print("\n--- Visualizing Poisoned Model Decision Boundaries ---")
 plot_decision_boundary_multi(
 X_train_poisoned, # Show poisoned points
 y_train_poisoned,
 poisoned_model, # Use the poisoned model for boundaries
 title=f"Poisoned Model Decision Boundary\nTarget Pred: {target_pred_poisoned} | Required: {MISCLASSIFY_AS_CLASS}",
 highlight_indices=[target_index] + perturbed_indices.tolist(),
 highlight_markers=["P"] + ["o"] * len(perturbed_indices),
 highlight_colors=[white] + [aquamarine] * len(perturbed_indices),
 highlight_labels=[f"Target (Pred: {target_pred_poisoned})"]
 + [f"Perturbed (Idx {idx})" for idx in perturbed_indices],
 )

else:
 print(
 "\nSkipping evaluation and visualization due to errors in attack implementation or model training."
 )

# Extract weights and intercept from the successfully trained poisoned model
if attack_logic_successful and poisoned_model is not None:
 try:
  # For OvR, we need weights/intercepts from all base estimators
  # The evaluator expects a list of lists for weights and a list for intercepts
  weights_list = [est.coef_[0].tolist() for est in poisoned_model.estimators_]
  intercept_list = [est.intercept_[0] for est in poisoned_model.estimators_]
  print("Extracted weights and intercepts from the poisoned model.")
  print(
  f"Weights shape (list of lists): ({len(weights_list)}, {len(weights_list[0]) if weights_list else 0})"
  )
  print(f"Intercept shape (list): ({len(intercept_list)})")
  submission_ready = True
 except Exception as e:
  print(
  f"{malware_red}Error:{white} Failed to extract parameters from poisoned model: {e}"
  )
  submission_ready = False
else:
 print(
 "Skipping parameter extraction because the model was not trained successfully."
 )
 submission_ready = False

# Replace <EVALUATOR_IP> and <PORT> with the correct values for the lab environment.
# Example: evaluator_base_url = "http://10.10.10.5:5000"
# evaluator_base_url = "http://<EVALUATOR_IP>:<PORT>"
evaluator_base_url = "http://94.237.61.248:51663"

# --- Health Check ---
health_check_url = f"{evaluator_base_url}/health"
print(f"Checking evaluator health at: {health_check_url}")

if "<EVALUATOR_IP>" in evaluator_base_url:
 print(f"\n{nugget_yellow}--- WARNING ---")
 print(
 "Please update the 'evaluator_base_url' variable with the correct IP and Port before running!"
 )
 print("----------------{white}")
else:
 try:
  response = requests.get(health_check_url, timeout=10)
  response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
  health_status = response.json()
  print("\n--- Health Check Response ---")
  print(f"Status: {health_status.get('status', 'N/A')}")
  print(f"Message: {health_status.get('message', 'No message received.')}")
  if health_status.get("status") != "healthy":
   print(
   f"\n{nugget_yellow}Warning:{white} Evaluator service reported an unhealthy status. Submission might fail."
   )
 except requests.exceptions.ConnectionError:
  print(
  f"\n{malware_red}Connection Error:{white} Could not connect to {health_check_url}."
  )
  print(
  "Please check the evaluator URL (IP/Port) and ensure the Docker container is running."
  )
 except requests.exceptions.Timeout:
  print(
  f"\n{malware_red}Timeout Error:{white} The request to {health_check_url} timed out."
  )
 except requests.exceptions.RequestException as e:
  print(f"\n{malware_red}Error during health check:{white} {e}")
 except json.JSONDecodeError:
  print(
  f"\n{malware_red}Error:{white} Could not decode JSON response from health check."
  )
  print(f"Raw status: {response.status_code}, Raw text: {response.text}")
 except Exception as e:
  print(
  f"\n{malware_red}An unexpected error occurred during health check:{white} {e}"
  )

# --- Submission ---
evaluator_url = f"{evaluator_base_url}/evaluate"
print(f"\nAttempting submission to: {evaluator_url}")

if "<EVALUATOR_IP>" in evaluator_base_url:
 print(f"\n{nugget_yellow}--- WARNING ---")
 print("Update 'evaluator_base_url' with the correct IP and Port before submitting!")
 print("----------------{white}")
elif not submission_ready:
 print(
 f"\n{malware_red}Submission aborted:{white} Model parameters were not extracted successfully."
 )
else:
 # Prepare payload
 payload = {"weights": weights_list, "intercept": intercept_list}
 # print(f"Payload preview: {json.dumps(payload, indent=2)[:500]}...")

 try:
  response = requests.post(evaluator_url, json=payload, timeout=30)
  response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
  result = response.json()

  print("\n--- Evaluator Response ---")
  if result.get("success"):
   print(f"{htb_green}Attack Successful!{white}")
   # Accuracy reported by server might be on the *poisoned* test set or a hidden one
   print(f"Accuracy reported by server: {result.get('accuracy', 'N/A'):.4f}")
   print(f"Target Point Check: {result.get('message', 'No message.')}")
   print(f"{htb_green}Flag: {result.get('flag', 'FLAG_NOT_RETURNED')}{white}")
  else:
   print(f"{malware_red}Evaluation Failed.{white}")
   accuracy_val = result.get("accuracy")
   accuracy_str = f"{accuracy_val:.4f}" if accuracy_val is not None else "N/A"
   print(f"Accuracy reported by server: {accuracy_str}")
   print(f"Message: {result.get('message', 'No specific message.')}")
   print("\nHints: Did your attack successfully misclassify the target point?")
   print(
   "Did you perturb the correct neighbors? Is epsilon_cross appropriate?"
   )
   print("Did you submit parameters for a 3-class OvR model?")

 except requests.exceptions.ConnectionError:
  print(
  f"\n{malware_red}Connection Error:{white} Could not connect to {evaluator_url}."
  )
  print("Check the evaluator URL and ensure the Docker container is running.")
 except requests.exceptions.Timeout:
  print(
  f"\n{malware_red}Timeout Error:{white} The request to {evaluator_url} timed out."
  )
 except requests.exceptions.RequestException as e:
  # This catches HTTPError (4xx, 5xx) as well
  print(f"\n{malware_red}Error during submission:{white} {e}")
  try:
   # Try to get more info from the response if available
   error_details = response.json()
   print(f"Server responded with: {error_details}")
  except (AttributeError, json.JSONDecodeError):
   if hasattr(response, "text"):
    print(f"Raw response content: {response.text}")
   else:
    print("(No response content available)")
 except json.JSONDecodeError:
  print(
  f"\n{malware_red}Error:{white} Could not decode JSON response from evaluator."
  )
  print(f"Raw status: {response.status_code}, Raw text: {response.text}")
 except Exception as e:
  print(
  f"\n{malware_red}An unexpected error occurred during submission:{white} {e}"
  )
```

We get the below output with the flag

```
Data loaded successfully from .npz file.
X_train shape: (1260, 2)
y_train shape: (1260,)
X_test shape: (540, 2)
y_test shape: (540,)
Target Point Index: 334
Target Point True Class (TARGET_CLASS): 2
Inferred Perturbing Class (PERTURBING_CLASS): 1
Required Misclassification Class (MISCLASSIFY_AS_CLASS): 1
Visualization functions defined.

--- Visualizing Clean Training Data with Target Point ---
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.

Attack function executed. Poisoned data shape: (1260, 2)
Indices perturbed: [1123  586 1214  555  596  785 1082 1231 1156  982 1122  385]

--- Training Final Model on Poisoned Data ---
Final model trained successfully on poisoned data.

--- Evaluating Poisoned Model ---
Target Point Evaluation (Index: 334):
 Original True Label: 2
 Poisoned Model Prediction: 1
 #9fef00Success:#ffffff The poisoned model misclassified the target point as the required Class 1.

Overall Performance on Clean Test Set (for info):
 Poisoned Model Accuracy: 0.9815

Classification Report (Poisoned Model on Clean Test Data):
              precision    recall  f1-score   support

     Class 0       1.00      1.00      1.00       180
     Class 1       0.97      0.98      0.97       180
     Class 2       0.98      0.97      0.97       180

    accuracy                           0.98       540
   macro avg       0.98      0.98      0.98       540
weighted avg       0.98      0.98      0.98       540


--- Visualizing Poisoned Training Data ---

--- Visualizing Poisoned Model Decision Boundaries ---
Extracted weights and intercepts from the poisoned model.
Weights shape (list of lists): (3, 2)
Intercept shape (list): (3)
Checking evaluator health at: http://94.237.61.248:51663/health

--- Health Check Response ---
Status: healthy
Message: Evaluator API is running.

Attempting submission to: http://94.237.61.248:51663/evaluate

--- Evaluator Response ---
#9fef00Attack Successful!#ffffff
Accuracy reported by server: 0.9815
Target Point Check: Attack successful! Target point (Index 334, True Class 2) misclassified as 1. Overall accuracy (0.9815) maintained and sensitivity checks passed.
#9fef00Flag: HTB{cl3an_l4b3l_fl4g_fun}#ffffff
```

