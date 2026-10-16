

Now that we have a baseline, we can proceed with the actual `Clean Label Attack`. Our specific goal is to modify the training data such that a chosen `target point`, 𝐱target, which originally belongs to `Class 1` (Yellow), will be misclassified by the retrained model as belonging to `Class 0` (Blue).

We aim to choose a point that genuinely belongs to `Class 1` (its true label ytarget=1) but also lies relatively close to the decision boundary separating `Class 1` from `Class 0`, as determined by the original baseline model. Points near the boundary are inherently more vulnerable to misclassification if the boundary shifts, even slightly, after retraining on the poisoned data.

To identify such a point, we can analyze the decision function scores produced by the baseline model. Remember that the decision boundary between `Class 0` and `Class 1` is where their respective scores, z0 and z1, are equal. We can define a function representing the difference between these scores:

f01(𝐱)=z0−z1=(𝐰0−𝐰1)T𝐱+(b0−b1)

The baseline model predicts `Class 1` for a point 𝐱 if its score z1 is greater than the scores for all other classes k. Specifically considering `Class 0` and `Class 1`, the model favors `Class 1` if z1>z0. This condition is equivalent to the score difference f01(𝐱) being negative (f01(𝐱)<0).

Therefore, we are looking for a specific point 𝐱target within the training set that meets our criteria: ytarget=1, and its score difference f01(𝐱target) must be negative (confirming the baseline model classifies it correctly relative to `Class 0`), while also being as close to zero as possible. A score difference that is the largest negative value indicates the point is correctly classified but is nearest to the f01(𝐱)=0 boundary.

To find this optimal target point, we calculate f01(𝐱) for all training points 𝐱i whose true label yi is 1. We then select the specific point 𝐱target that yields the largest negative value (i.e., the value closest to zero) for f01.

Code: python

```python
print("\n--- Selecting Target Point ---")
# We use the baseline parameters w0_base, b0_base, w1_base, b1_base extracted earlier
# Calculate the difference vector and intercept for the 0-vs-1 boundary
w_diff_01_base = w0_base - w1_base
b_diff_01_base = b0_base - b1_base
print(f"Boundary vector (w0-w1): {w_diff_01_base}")
print(f"Intercept difference (b0-b1): {b_diff_01_base}")

# Identify indices of all Class 1 points in the original clean training set
class1_indices_train = np.where(y_train_3c == 1)[0]

if len(class1_indices_train) == 0:
    raise ValueError(
        "CRITICAL: No Class 1 points found in the training data. Cannot select target."
    )
else:
    print(f"Found {len(class1_indices_train)} Class 1 points in the training set.")

# Get the feature vectors for only the Class 1 points
X_class1_train = X_train_3c[class1_indices_train]

# Calculate the decision function f_01(x) = (w0-w1)^T x + (b0-b1) for these Class 1 points
# A negative value means the point is on the Class 1 side of the 0-vs-1 boundary
decision_values_01 = X_class1_train @ w_diff_01_base + b_diff_01_base

# Find indices within the subset of Class 1 points that are correctly classified (f_01 < 0)
class1_on_correct_side_indices_relative = np.where(decision_values_01 < 0)[0]

if len(class1_on_correct_side_indices_relative) == 0:
    # This case is unlikely if the baseline model has decent accuracy, but handle it.
    print(
        f"{malware_red}Warning:{white} No Class 1 points found on the expected side (f_01 < 0) of the 0-vs-1 baseline boundary."
    )
    print(
        "Selecting the Class 1 point with the minimum absolute decision value instead."
    )
    # Find index (relative to class1 subset) with the smallest absolute distance to boundary
    target_point_index_relative = np.argmin(np.abs(decision_values_01))
else:
    # Among the correctly classified points, find the one closest to the boundary
    # This corresponds to the maximum (least negative) decision value
    target_point_index_relative = class1_on_correct_side_indices_relative[
        np.argmax(decision_values_01[class1_on_correct_side_indices_relative])
    ]

# Map the relative index (within the class1 subset) back to the absolute index in the original X_train_3c array
target_point_index_absolute = class1_indices_train[target_point_index_relative]

# Retrieve the target point's features and true label
X_target = X_train_3c[target_point_index_absolute]
y_target = y_train_3c[
    target_point_index_absolute
]  # Should be 1 based on selection logic

# Sanity Check: Verify the chosen point's class and baseline prediction
target_baseline_pred = baseline_model_3c.predict(X_target.reshape(1, -1))[0]
target_decision_value = decision_values_01[target_point_index_relative]

print(f"\nSelected Target Point Index (absolute): {target_point_index_absolute}")
print(f"Target Point Features: {X_target}")
print(f"Target Point True Label (y_target): {y_target}")
print(f"Target Point Baseline Prediction: {target_baseline_pred}")
print(
    f"Target Point Baseline 0-vs-1 Decision Value (f_01): {target_decision_value:.4f}"
)

if y_target != 1:
    print(
        f"Error: Selected target point does not have label 1! Check logic."
    )
if target_baseline_pred != y_target:
    print(
        f"Warning: Baseline model actually misclassifies the chosen target point ({target_baseline_pred}). Attack might trivially succeed or have unexpected effects."
    )
if target_decision_value >= 0:
    print(
        f"Warning: Selected target point has f_01 >= 0 ({target_decision_value:.4f}), meaning it wasn't on the Class 1 side of the 0-vs-1 boundary. Check logic or baseline model."
    )

# Visualize the data highlighting the selected target point near the boundary
print("\n--- Visualizing Training Data with Target Point ---")
plot_data_multi(
    X_train_3c,
    y_train_3c,
    title="Training Data Highlighting the Target Point (Near Boundary)",
    highlight_indices=[target_point_index_absolute],
    highlight_markers=["P"],  # 'P' for Plus sign marker (Target)
    highlight_colors=[white],  # White edge color for visibility
    highlight_labels=[f"Target (Class {y_target}, Idx {target_point_index_absolute})"],
)
```

The above code identifies a good candidate for us to work with, index `373`:

Code: python

```python
--- Selecting Target Point ---
Boundary vector (w0-w1): [-5.78792514  6.32142485]
Intercept difference (b0-b1): -0.9207223376477074
Found 350 Class 1 points in the training set.

Selected Target Point Index (absolute): 373
Target Point Features: [-0.55111155 -0.36675028]
Target Point True Label (y_target): 1
Target Point Baseline Prediction: 1
Target Point Baseline 0-vs-1 Decision Value (f_01): -0.0493

```

If we plot index `373` we can easily see where it is in the dataset.

![Scatter plot titled 'Training Data Highlighting the Target Point (Near Boundary)' showing Class 0 (blue), Class 1 (orange), Class 2 (red), and Target Point (white cross) across standardized Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/feature_attack_targted_point.png)

