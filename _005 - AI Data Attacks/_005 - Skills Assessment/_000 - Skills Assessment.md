

Your objective in this assessment is to strategically manipulate a model's training data to achieve a specific, nuanced misclassification behavior.

You will be provided a dataset in the zip attached to the question, along with a template notebook.

You will target a 4-class `One-vs-Rest` (`OvR`) `Logistic Regression` classifier. The goal is to poison its training dataset using only label flipping such that the subsequently trained model exhibits ambiguous classification for instances of `Class 1`. In other words, when the poisoned model encounters new data points that genuinely belong to `Class 1`, it should frequently misclassify them as either `Class 0` or `Class 2`, degrading the classification accuracy of `Class 1`.

Implement your attack strategy within the provided template notebook and use the same notebook to submit your poisoned model to the instancea api for evaluation.

1. Submit the flag you receive from a successful attack as the answer to this question.

Execute the below script after extracting the file `assessment_dataset.npz`

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import requests
import os

# Load the dataset
data = np.load('assessment_dataset.npz')
X_train_orig = data['X_train']
y_train_orig = data['y_train']
X_test = data['X_test']  # Optional, for local testing
y_test = data['y_test']  # Optional, for local testing

# Set SEED if not defined
SEED = 1337

# Poisoning strategy: Flip a portion of Class 1 samples to Class 0 and Class 2
# Specifically, flip the closest ones to the centroids of 0 and 2
# Use a fraction to ensure some Class 1 remains (to keep 4 classes)

y_train_poisoned = y_train_orig.copy()

# Compute centroids for Class 0 and Class 2
cent0 = np.mean(X_train_orig[y_train_orig == 0], axis=0)
cent2 = np.mean(X_train_orig[y_train_orig == 2], axis=0)

# Get Class 1 instances
class1_mask = y_train_orig == 1
class1_indices = np.where(class1_mask)[0]
class1_X = X_train_orig[class1_mask]

# Distances to centroids
dists_to_0 = np.linalg.norm(class1_X - cent0, axis=1)
dists_to_2 = np.linalg.norm(class1_X - cent2, axis=1)

# Sort by distance
sorted_idx_0 = np.argsort(dists_to_0)
sorted_idx_2 = np.argsort(dists_to_2)

# Flip fraction (adjust if needed; 0.35 to each side -> ~70% flipped, 30% remain)
flip_fraction = 0.35
num_flip = int(len(class1_indices) * flip_fraction)

# Flip closest to Class 0
flip_to_0 = class1_indices[sorted_idx_0[:num_flip]]
y_train_poisoned[flip_to_0] = 0

# Flip closest to Class 2 (may have some overlap, but last assignment wins)
flip_to_2 = class1_indices[sorted_idx_2[:num_flip]]
y_train_poisoned[flip_to_2] = 2

# Verify unique classes (should be 4)
print("Unique classes after poisoning:", np.unique(y_train_poisoned))
print("Label distribution after poisoning:", np.bincount(y_train_poisoned))

# Train the poisoned model with the exact config
base_estimator_config = {
    "random_state": SEED,
    "solver": "liblinear",
    "C": 1.0,
    "max_iter": 200,
}
poisoned_estimator = LogisticRegression(**base_estimator_config)
poisoned_model = OneVsRestClassifier(poisoned_estimator)
poisoned_model.fit(X_train_orig, y_train_poisoned)

# Save parameters to npz as required
output_model_filename = "poisoned_model_params.npz"
params_to_save = {}
for i, estimator in enumerate(poisoned_model.estimators_):
    params_to_save[f"coef_estimator_{i}"] = estimator.coef_
    params_to_save[f"intercept_estimator_{i}"] = estimator.intercept_
params_to_save["classes_"] = poisoned_model.classes_
np.savez_compressed(output_model_filename, **params_to_save)

print(f"Poisoned model parameters saved to {output_model_filename}")

# Submit to the API
API_URL = "http://83.136.255.53:37089/evaluate_model"
if os.path.exists(output_model_filename):
    with open(output_model_filename, "rb") as f:
        files = {"model_params": (output_model_filename, f, "application/octet-stream")}
        response = requests.post(API_URL, files=files)
    print("API Response:")
    print(response.text)
else:
    print("Model file not found.")
```

The output below contains the flag

```
Unique classes after poisoning: [0 1 2 3]
Label distribution after poisoning: [459 160 506 375]
Poisoned model parameters saved to poisoned_model_params.npz
API Response:
{"attack_successful":true,"criteria_met":{"class1_to_class0_ambiguity":true,"class1_to_class2_ambiguity":true,"class3_recall_maintained":true},"evaluation_status":"complete","flag":"HTB{4mbiguity_m4st3r}","message":"Attack successful! Ambiguity achieved and constraints met.","metrics":{"class1_correctly_as_1":15,"class1_misclassified_as_0":35,"class1_misclassified_as_0_percent":0.28,"class1_misclassified_as_2":64,"class1_misclassified_as_2_percent":0.512,"class1_misclassified_as_3":11,"class1_total_samples":125,"class3_precision":0.9185185185185185,"class3_recall":0.992,"overall_accuracy":0.778}}
```

