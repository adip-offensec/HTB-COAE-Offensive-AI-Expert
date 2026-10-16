

Having identified the `target point` 𝐱target, our next step is to manipulate the training data specifically to cause its misclassification. We achieve this by subtly shifting the learned `decision boundary`. We will perturb the selected `Class 0` (Blue) data points that are neighbours to the `target point` in order to shift the boundary.

We first need to locate several `Class 0` points residing closest to 𝐱target within the feature space. These neighbours serve as anchors influencing the boundary’s local position. We then calculate small `perturbations`, denoted δi, for these selected neighbours 𝐱i. These perturbations are specifically designed to push each neighbour slightly across the original decision boundary (f01(𝐱)=0) and into the region typically associated with `Class 1` (Yellow). This process yields perturbed points 𝐱′i=𝐱i+δi.

The poisoned training dataset is then created by substituting these original neighbours 𝐱i with their perturbed counterparts 𝐱′i. Crucially, we assign the original `Class 0` label to these perturbed points 𝐱′i, even though they now sit in the `Class 1` region according to the baseline model.

When the model retrains on this poisoned data, it encounters a conflict: points (𝐱′i) labeled `0` are located where it would expect points labeled `1`. To reconcile this based on the provided (and unchanged) labels, the model is forced to adjust its decision boundary. Typically, it pushes the boundary f01(𝐱)=0 outwards into the original `Class 1` region to correctly classify the perturbed points 𝐱′i as `Class 0`. A successful attack occurs when this induced boundary shift is significant enough to engulf the nearby `target point` 𝐱target, causing it to fall on the `Class 0` side of the new boundary.

Throughout this process, the `perturbations` δi must remain small. This subtlety ensures the `Class 0` label still appears plausible for the altered feature vectors 𝐱′i, thus preserving the "clean label" characteristic of the attack where only features are modified, not labels.

## Attack Implementation

We begin the implementation by finding the required `Class 0` neighbours closest to the `target point`. We use Scikit-learn’s `NearestNeighbors` algorithm, fitting it only on the `Class 0` training data and then querying it with the coordinates of 𝐱target. We must specify how many neighbours (`n_neighbors_to_perturb`) to select for modification.

Code: python

```python
print("\n--- Identifying Class 0 Neighbors to Perturb ---")
n_neighbors_to_perturb = 5 # Hyperparameter: How many neighbors to modify

# Find indices of all Class 0 points in the original training set
class0_indices_train = np.where(y_train_3c == 0)[0]

if len(class0_indices_train) == 0:
    raise ValueError("CRITICAL: No Class 0 points found. Cannot find neighbors to perturb.")
else:
    print(f"Found {len(class0_indices_train)} Class 0 points in the training set.")

# Get features of only Class 0 points
X_class0_train = X_train_3c[class0_indices_train]

# Sanity check to ensure we don't request more neighbors than available
if n_neighbors_to_perturb > len(X_class0_train):
    print(f"Warning: Requested {n_neighbors_to_perturb} neighbors, but only {len(X_class0_train)} Class 0 points available. Using all available.")
    n_neighbors_to_perturb = len(X_class0_train)

if n_neighbors_to_perturb == 0:
    raise ValueError("No Class 0 neighbors can be selected to perturb (n_neighbors_to_perturb=0). Cannot proceed.")

# Initialize and fit NearestNeighbors on the Class 0 data points
# We use the default Euclidean distance ('minkowski' with p=2)
nn_finder = NearestNeighbors(n_neighbors=n_neighbors_to_perturb, algorithm='auto')
nn_finder.fit(X_class0_train)

# Find the indices (relative to X_class0_train) and distances of the k nearest Class 0 neighbors to X_target
distances, indices_relative = nn_finder.kneighbors(X_target.reshape(1, -1))

# Map the relative indices found within X_class0_train back to the original indices in X_train_3c
neighbor_indices_absolute = class0_indices_train[indices_relative.flatten()]
# Get the original feature vectors of these neighbors (needed for perturbation)
X_neighbors = X_train_3c[neighbor_indices_absolute]

# Output the findings for verification
print(f"\nTarget Point Index: {target_point_index_absolute} (True Class {y_target})")
print(f"Identified {len(neighbor_indices_absolute)} closest Class 0 neighbors to perturb:")
print(f"  Indices in X_train_3c: {neighbor_indices_absolute}")
print(f"  Distances to target: {distances.flatten()}")

# Sanity check: Ensure the target itself wasn't accidentally included (e.g., if it was mislabeled or data is unusual)
if target_point_index_absolute in neighbor_indices_absolute:
     print(f"Error: The target point itself was selected as one of its own Class 0 neighbors. This indicates a potential issue in data or logic.")
```

This has identified the 5 closest Class 0 points to our target:

Code: python

```python
--- Identifying Class 0 Neighbors to Perturb ---
Found 350 Class 0 points in the training set.

Target Point Index: 373 (True Class 1)
Identified 5 closest Class 0 neighbors to perturb:
  Indices in X_train_3c: [ 761   82 1035  919  491]
  Distances to target: [0.10318016 0.12277741 0.14917583 0.25081115 0.30161621]
```

Having identified the neighbours, we now determine the exact change (`perturbation`) to apply to each one. The goal is to push these points (𝐱i) from their original `Class 0` region (where f01(𝐱i)>0) just across the boundary into the `Class 1` region (where f01<0).

The most direct path across the boundary f01(𝐱)=(𝐰0−𝐰1)T𝐱+(b0−b1)=0 is perpendicular to it. The vector 𝐯01=(𝐰0−𝐰1), which defines the boundary, is the `normal vector` and points perpendicular to the boundary hyperplane. To move a point from the `Class 0` side to the `Class 1` side, we need to push it in the direction `opposite` to this normal vector, namely −𝐯01.

We first normalize this direction vector to obtain a unit vector indicating the push direction:

𝐮push=−𝐯01∥𝐯01∥=−(𝐰0−𝐰1)∥𝐰0−𝐰1∥

The distance we push the points is controlled by a small hyperparameter, ϵcross. This value determines how far across the boundary the neighbours are shifted. Smaller values yield subtler changes, while larger values create a stronger push but might make the perturbed points less plausible as `Class 0`.

The final perturbation vector δi applied to each neighbour 𝐱i is the unit push direction scaled by the chosen magnitude:

δi=ϵcross×𝐮push

Applying this results in the perturbed point 𝐱′i=𝐱i+δi. We expect the original neighbour 𝐱i to satisfy f01(𝐱i)>0, while the perturbed point 𝐱′i should satisfy f01(𝐱′i)<0.

Code: python

```python
print("\n--- Calculating Perturbation Vector ---")
# Use the boundary vector w_diff_01_base = w0_base - w1_base calculated earlier

# The direction to push Class 0 points into Class 1 region is opposite to the normal vector (w0-w1)
push_direction = -w_diff_01_base
norm_push_direction = np.linalg.norm(push_direction)

# Handle potential zero vector for the boundary normal
if norm_push_direction < 1e-9: # Use a small threshold for floating point comparison
    raise ValueError("Boundary vector norm (||w0-w1||) is close to zero. Cannot determine push direction reliably.")
else:
    # Normalize the direction vector to unit length
    unit_push_direction = push_direction / norm_push_direction
    print(f"Calculated unit push direction vector (normalized - (w0-w1)): {unit_push_direction}")

# Define perturbation magnitude (how far across the boundary to push)
epsilon_cross = 0.25
print(f"Perturbation magnitude (epsilon_cross): {epsilon_cross}")

# Calculate the final perturbation vector (direction * magnitude)
perturbation_vector = epsilon_cross * unit_push_direction
print(f"Final perturbation vector (delta): {perturbation_vector}")
```

With this, we have calculated the vector to apply:

Code: python

```python
--- Calculating Perturbation Vector ---
Calculated unit push direction vector (normalized - (w0-w1)): [ 0.67529883 -0.73754423]
Perturbation magnitude (epsilon_cross): 0.25
Final perturbation vector (delta): [ 0.16882471 -0.18438606]
```

We apply this single calculated `perturbation_vector` to each of the selected `Class 0` neighbors to generate the poisoned dataset. We begin by creating a `safe copy` of the original training features and labels, named `X_train_poisoned` and `y_train_poisoned` respectively. Then, we iterate through the indices of the identified neighbours (`neighbor_indices_absolute`). For each `neighbor_idx`, we retrieve its original feature vector 𝐱i, calculate the perturbed vector 𝐱′i=𝐱i+perturbation_vector, and update the corresponding entry in `X_train_poisoned`. The label for this index in `y_train_poisoned` remains unchanged as `0`, copied from the original `y_train_3c`. This loop constructs the final poisoned dataset ready for retraining.

Code: python

```python
print("\n--- Applying Perturbations to Create Poisoned Dataset ---")
# Create a safe copy of the original training data to modify
X_train_poisoned = X_train_3c.copy()
y_train_poisoned = (
    y_train_3c.copy()
)  # Labels are copied but not changed for perturbed points

perturbed_indices_list = []  # Keep track of which indices were actually modified

# Iterate through the identified neighbor indices and their original features
# neighbor_indices_absolute holds the indices in X_train_3c/y_train_3c
# X_neighbors holds the corresponding original feature vectors
for i, neighbor_idx in enumerate(neighbor_indices_absolute):
    X_neighbor_original = X_neighbors[i]  # Original feature vector of the i-th neighbor

    # Calculate the new position of the perturbed neighbor
    X_perturbed_neighbor = X_neighbor_original + perturbation_vector

    # Replace the original neighbor's features with the perturbed features in the copied dataset
    X_train_poisoned[neighbor_idx] = X_perturbed_neighbor
    # The label y_train_poisoned[neighbor_idx] remains 0 (Class 0)

    perturbed_indices_list.append(neighbor_idx)  # Record the index that was modified

    # Verify the effect of perturbation on the f_01 score
    f01_orig = X_neighbor_original @ w_diff_01_base + b_diff_01_base
    f01_pert = X_perturbed_neighbor @ w_diff_01_base + b_diff_01_base
    print(f"  Neighbor Index {neighbor_idx} (Label 0): Perturbed.")
    print(
        f"     Original f01 = {f01_orig:.4f} (>0 expected), Perturbed f01 = {f01_pert:.4f} (<0 expected)"
    )
    if f01_pert >= 0:
        print(
            f"     Warning: Perturbed point did not cross the baseline boundary (f01 >= 0). Epsilon might be too small."
        )

print(
    f"\nCreated poisoned training dataset by perturbing features of {len(perturbed_indices_list)} Class 0 points."
)
# Check the size to ensure it's unchanged
print(
    f"Poisoned training dataset size: {X_train_poisoned.shape[0]} samples (should match original {X_train_3c.shape[0]})."
)

# Convert list to numpy array for potential use later
perturbed_indices_arr = np.array(perturbed_indices_list)

# Final safety check: ensure target wasn't modified
if target_point_index_absolute in perturbed_indices_arr:
    print(
        f"CRITICAL Error: Target point index {target_point_index_absolute} was included in the perturbed indices! Check neighbor finding logic."
    )

# Visualize the poisoned dataset, highlighting target and perturbed points
print("\n--- Visualizing Poisoned Training Data ---")
plot_data_multi(
    X_train_poisoned,  # Use the poisoned features
    y_train_poisoned,  # Use the corresponding labels (perturbed points still have label 0)
    title="Poisoned Training Data (Features Perturbed)",
    highlight_indices=[target_point_index_absolute] + perturbed_indices_list,
    highlight_markers=["P"]
    + ["o"]
    * len(perturbed_indices_list),  # 'P' for Target, 'o' for Perturbed neighbors
    highlight_colors=[white]
    + [vivid_purple]
    * len(perturbed_indices_list),  # White edge Target, Purple edge Perturbed
    highlight_labels=[f"Target (Idx {target_point_index_absolute}, Class {y_target})"]
    + [f"Perturbed (Idx {idx}, Label 0)" for idx in perturbed_indices_list],
)
```

The plot below shows the `poisoned training dataset`. The `target point` ('+') remains unchanged in `Class 1`. The `perturbed neighbors` (points with purple edges) started as `Class 0` points (Azure) near the target but have been shifted slightly into the `Class 1` region (Yellow). This visual discrepancy - blue points in the yellow region - is what forces the model to adjust its boundary during training.

![Scatter plot titled 'Poisoned Training Data (Features Perturbed)' showing Class 0 (blue), Class 1 (orange), Class 2 (red), Target Point (white cross), and Perturbed Points (purple) across standardized Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/feature_attack_perturbed.png)

