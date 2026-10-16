

In this section, we will train a random forest model on the NSL-KDD dataset for multi-class classification. The goal is to build a model that can accurately classify network traffic into different attack categories or as normal traffic.

## Training the Model

Code: python

```python
# Train RandomForest model for multi-class classification
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

The first step in this process is to train the random forest model using the training subset of the dataset. We initialize a `RandomForestClassifier` with the `random_state` parameter set to `1337` to ensure reproducibility. The `fit` method is then used to train the model on the features `multi_train_X` and the target variable `multi_train_y`. This step builds the model by learning patterns from the training data.

## Evaluating the Model on the Validation Set

Next, we will evaluate the performance of the trained random forest model on the validation set. The goal is to assess the model's accuracy and other performance metrics to ensure it generalizes well to unseen data.

Code: python

```python
# Predict and evaluate the model on the validation set
multi_predictions = rf_model_multi.predict(multi_val_X)
accuracy = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')
print(f"Validation Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix for Validation Set
conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Validation Set
print("Classification Report for Validation Set:")
print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```

After training the model, we use it to make predictions on the validation set. The `predict` method of the `RandomForestClassifier` is used to generate predictions for the features `multi_val_X`. We then calculate various performance metrics using functions from `sklearn.metrics`:

- `Accuracy`: The proportion of correctly classified instances.
- `Precision`: The ratio of true positive predictions to the total predicted positives.
- `Recall`: The ratio of true positive predictions to the total actual positives.
- `F1-Score`: The harmonic mean of precision and recall.

These metrics are printed to evaluate the model's performance on the validation set comprehensively.

We also generate a confusion matrix using `confusion_matrix` and visualize it using `seaborn` and `matplotlib`. The confusion matrix provides a detailed breakdown of the model's predictions, showing each class's number of true positives, true negatives, false positives, and false negatives.

Finally, we print a classification report that includes precision, recall, F1-score, and support for each class. This report gives a more granular view of the model's performance across different classes.

## Testing the Model on the Test Set

![Confusion matrix for network anomaly detection: 15,349 normal, 10,708 DoS, 2,788 probe, 703 access, with minor misclassifications.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/anomaly_test.png)

Next, we will evaluate the final performance of the trained random forest model on the test set. The goal is to assess the model's ability to generalize to completely unseen data and provide a final evaluation of its performance.

Code: python

```python
# Final evaluation on the test set
test_multi_predictions = rf_model_multi.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Confusion Matrix for Test Set
test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Test Set
print("Classification Report for Test Set:")
print(classification_report(test_y, test_multi_predictions, target_names=class_labels))
```

The final step in our process is to evaluate the model on the test set. We use the `predict` method to generate predictions for the features `test_X`. Similar to the validation set evaluation, we calculate and print various performance metrics:

- `Accuracy`: The proportion of correctly classified instances.
- `Precision`: The ratio of true positive predictions to the total predicted positives.
- `Recall`: The ratio of true positive predictions to the total actual positives.
- `F1-Score`: The harmonic mean of precision and recall.

We also generate a confusion matrix for the test set and visualize it using `seaborn` and `matplotlib`. This matrix provides a detailed breakdown of the model's predictions on the test data, showing each class's number of true positives, true negatives, false positives, and false negatives.

Finally, we print a classification report that includes precision, recall, F1-score, and support for each class. This report gives a comprehensive view of the model's performance across different classes on the test set.

By executing this code, we have trained a random forest model, evaluated its performance on both the validation and test sets, and generated detailed reports and visualizations to assess its effectiveness in classifying network traffic.

## Saving Model

Save your model using this code:

Code: python

```python
import joblib

# Save the trained model to a file
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)

print(f"Model saved to {model_filename}")
```

