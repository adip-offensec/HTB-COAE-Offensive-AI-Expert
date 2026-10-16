

# Training and Deploying the Student Model

With pseudo-labeled public data generated through noisy aggregation, we now train the student model that will be deployed for inference. The student never accesses the sensitive training data directly, yet learns to make accurate predictions through knowledge distilled from the teacher ensemble.

## The Knowledge Distillation Paradigm

We can view PATE as a form of `knowledge distillation` where the student learns from teacher predictions instead of raw data. In standard knowledge distillation, a smaller student model learns to mimic a larger teacher model's outputs. PATE extends this idea to an ensemble of teachers, with the addition of noise for privacy.

Standard knowledge distillation transfers `soft probability distributions` from teacher to student, providing richer training signal than hard labels. PATE uses `hard pseudo-labels` (the argmax of noisy votes) because soft probabilities would leak more information about teacher confidences and thus more information about the training data. Despite using only hard labels, PATE works well because the label captures the essential prediction.

What benefits does this paradigm provide beyond privacy? The student model has the same architecture as one teacher but captures the ensemble's collective knowledge. Learning from pseudo-labels also acts as implicit regularization, potentially improving generalization compared to learning from raw labels. Finally, the resulting student is a standard neural network with no special inference requirements, making deployment straightforward.

## Training the PATE Student

The student model trains on the pseudo-labeled data generated in the previous phase. Because the student never touches sensitive data, we can use the same architecture as teachers and train without privacy constraints:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 4: Training Student Model")
print("=" * 60)

student_model = MLP(
    input_size=num_features,
    hidden_layers=STUDENT_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=STUDENT_CONFIG['dropout']
)

student_train_loader = create_dataloader(
    X_query, student_labels,
    STUDENT_CONFIG['batch_size']
)
```

Why use the same `[128, 64]` architecture as teachers? Knowledge distillation works best when student capacity matches the complexity of teacher predictions. A larger student might overfit the noisy labels; a smaller one might underperform. Because training uses only public data with derived labels, no privacy budget is consumed. We could train for 100 epochs or restart training entirely without affecting the privacy guarantee established during the labeling phase.

#### Standard Training Loop

We train the student using standard supervised learning with cross-entropy loss, using the noisy pseudo-labels as ground truth and treating them no differently than human-provided labels:

Code: python

```python
train_model(
    student_model, student_train_loader, holdout_loader,
    device=DEVICE,
    epochs=STUDENT_CONFIG['epochs'],
    learning_rate=STUDENT_CONFIG['learning_rate']
)

student_test_acc = evaluate_accuracy(student_model, holdout_loader, DEVICE)
print(f"\nPATE student test accuracy: {student_test_acc:.2f}%")
```

After 30 epochs, we evaluate on the holdout set to measure true generalization. The student typically achieves around 88% accuracy, demonstrating that PATE preserves model utility despite the noisy labeling process.

## Understanding Label Noise Resilience

How does the student learn effectively despite noisy labels? Two factors explain this resilience: the concentration of noise on ambiguous samples, and the inherent robustness of neural network training to moderate label corruption.

#### Noise Distribution Across Samples

High-consensus samples receive correct labels almost always because the winning class has a large margin over alternatives. Only samples where teachers genuinely disagree experience label flips from noise. This non-uniform noise distribution means the student receives mostly clean supervision on clear-cut cases while receiving noisy supervision on genuinely ambiguous cases. The clear-cut examples dominate learning because they provide consistent gradient signals.

#### Neural Network Robustness

Research on learning with noisy labels shows that neural networks trained with stochastic gradient descent exhibit natural robustness to label noise, particularly when that noise is random instead of systematic. Networks can learn correct patterns even when 20-40% of labels are corrupted, as long as the corruption is independent of the true labels.

PATE's noise satisfies this condition because it depends on random Laplacian draws, not systematic data properties. A sample does not receive a wrong label because of its features; it receives a wrong label because random noise happened to flip the vote. This randomness means the model cannot learn to exploit the noise pattern.

## Comparing with Clean Training

To understand the accuracy cost of privacy, we compare student performance against the clean ensemble that never added noise:

Code: python

```python
# Compare against ensemble performance
clean_preds = votes.argmax(axis=1)
clean_ensemble_acc = (clean_preds == y_query_true).mean() * 100
accuracy_gap = clean_ensemble_acc - student_test_acc

print(f"\nComparison:")
print(f"  Clean ensemble accuracy: {clean_ensemble_acc:.2f}%")
print(f"  PATE student accuracy:   {student_test_acc:.2f}%")
print(f"  Accuracy gap:            {accuracy_gap:.2f}%")
```

Typical results show the clean ensemble achieving approximately 92% accuracy (teachers voting without noise) while the PATE student achieves approximately 88% accuracy (trained on noisy labels). This modest 4-point gap represents the privacy cost, a favorable tradeoff for the formal differential privacy guarantees obtained.

## Model Deployment

After training, we deploy the PATE student without further privacy considerations. The privacy budget was entirely consumed during the labeling phase.

#### Unlimited Inference

Unlike techniques that add noise during inference (such as randomized response), we can query the PATE student unlimited times without additional privacy cost. Every query is free because the student's weights contain only knowledge distilled through the noisy labeling process. No private data flows during inference.

Code: python

```python
def deploy_model(model, new_data, scaler, device):
    """
    Deploy the PATE student for inference.

    No privacy budget consumed per query.
    """
    model.eval()
    X_norm = scaler.transform(new_data)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    return predictions
```

Now we demonstrate deployment with a batch of holdout samples:

Code: python

```python
# Example deployment
sample_data = X_holdout[:100]
predictions = deploy_model(student_model, sample_data, scaler, DEVICE)
print(f"Processed {len(predictions)} queries with no privacy cost")
```

How does deployment work? We normalize incoming data with the same scaler fitted on private data (preserving identical feature scaling), pass the tensor through the trained model, and extract class predictions via argmax. We could process millions of queries this way without consuming any privacy budget. This unlimited inference property makes PATE particularly valuable for high-traffic web services, batch processing pipelines, and digit recognition systems serving many users.

#### Saving the Model

We save the trained student for deployment using the `safetensors` format, which is safer and faster than pickle-based formats:

Code: python

```python
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

save_file(student_model.state_dict(), os.path.join(MODEL_DIR, "pate_student.safetensors"))
print(f"\nModel saved to {MODEL_DIR}/pate_student.safetensors")
```

The saved file contains only the student's learned weights. No information about the teacher ensemble or private training images remains. Loading this model in production requires only the architecture definition and the saved state dictionary.

