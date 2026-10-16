

# Data Partitioning and Teacher Training

Teacher quality determines student quality. Every label the student learns from originates in teacher predictions, so the first phase of PATE, partitioning data and training the teacher ensemble, directly impacts final model accuracy. Each teacher must learn meaningful patterns from its limited data partition while maintaining enough diversity across the ensemble to benefit from aggregation.

## Loading and Preparing MNIST

We load the `MNIST` dataset introduced in the previous section. The dataset's 60,000 training images will be partitioned for our three-way data split:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 1: Loading MNIST Dataset")
print("=" * 60)

train_loader, test_loader = get_mnist_loaders()

X_train = train_loader.dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_loader.dataset.targets.numpy()
X_test = test_loader.dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test = test_loader.dataset.targets.numpy()

num_features = 784
num_classes = 10
DATASET_CONFIG['num_features'] = num_features
DATASET_CONFIG['num_classes'] = num_classes

print(f"Dataset: {X_train.shape[0]} training samples, {num_features} features, {num_classes} classes")
print(f"Class distribution: min={np.bincount(y_train).min()}, max={np.bincount(y_train).max()}")
```

We use `get_mnist_loaders()` from the library to handle downloading and caching, then extract raw numpy arrays from the underlying datasets. Pixel values are normalized to [0, 1] by dividing by 255, and we flatten each 28x28 image into a 784-dimensional vector. Classes are well-balanced across all 10 digits, with approximately 6,000 samples per class. This balance matters for teacher training since each teacher will see roughly equal representation of all digits despite having a small partition of the data.

#### Creating the Three-Way Data Split

Our experimental design requires three distinct data sets with clean separation between training and evaluation. We use the standard `MNIST` train/test split as a starting point, then further partition the training data:

Code: python

```python
# Split training data into: private (for teachers) and public (for student queries)
X_private, X_public, y_private, y_public = train_test_split(
    X_train, y_train, test_size=0.2,
    random_state=RANDOM_SEED, stratify=y_train
)

# Use standard test set as holdout for evaluation
X_holdout = X_test
y_holdout = y_test

print(f"\nData splits:")
print(f"  Private (teacher training): {len(X_private)} samples")
print(f"  Public (student queries):   {len(X_public)} samples")
print(f"  Holdout (evaluation):       {len(X_holdout)} samples")
```

The first split takes 80% of the training data as private, leaving 20% as public data for student labeling queries. The standard `MNIST` test set (10,000 samples) serves as our holdout for final evaluation. With 60,000 total training samples, this yields approximately 48,000 private samples for teacher training, 12,000 public samples available for student labeling queries, and 10,000 holdout samples for final evaluation.

We use `stratify=y_train` to maintain the original class distribution across all 10 digit classes in each split. Without stratification, random chance could produce splits where some digits are underrepresented, which would affect teacher training quality.

#### Feature Normalization

Neural networks train more effectively when features have similar scales. We fit the scaler on private data only, then apply the same transformation to the other sets:

Code: python

```python
scaler = StandardScaler()
X_private_norm = scaler.fit_transform(X_private)
X_public_norm = scaler.transform(X_public)
X_holdout_norm = scaler.transform(X_holdout)
```

Fitting only on private data mirrors a realistic deployment scenario where normalization parameters come from the sensitive data source. Using `transform` (not `fit_transform`) on public and holdout data ensures these sets undergo identical scaling without leaking information about their distributions back into the normalization.

## Creating Disjoint Partitions

We partition the private training data into disjoint subsets, with each teacher receiving approximately equal samples:

Code: python

```python
print("\n" + "=" * 60)
print("PHASE 2: Training Teacher Ensemble")
print("=" * 60)

num_teachers = TEACHER_CONFIG['num_teachers']
```

With the phase banner displayed, we create the partitions by shuffling indices and slicing into equal chunks:

Code: python

```python
# Partition private data for teachers
np.random.seed(RANDOM_SEED)
indices = np.random.permutation(len(X_private_norm))
partition_size = len(X_private_norm) // num_teachers

teacher_partitions = []
for i in range(num_teachers):
    start_idx = i * partition_size
    if i == num_teachers - 1:
        partition_indices = indices[start_idx:]
    else:
        partition_indices = indices[start_idx:start_idx + partition_size]
    teacher_partitions.append(partition_indices)

print(f"Created {num_teachers} partitions, ~{partition_size} samples each")
```

Shuffling before partitioning prevents unintended clustering. If we sliced the original dataset in order, samples might cluster by some hidden property (collection date, source, etc.). Random permutation ensures each partition samples uniformly from the full distribution. We slice this shuffled array into 250 consecutive chunks of approximately 192 samples each. When 48,000 does not divide evenly by 250, the final partition absorbs the remainder (48,000 mod 250 = 0 in this case, so partitions are exactly equal).

This disjoint structure is essential for PATE's privacy guarantee. If partitions overlapped, a single training sample could influence multiple teachers, amplifying information leakage when that sample's membership is queried. With strictly disjoint partitions, each sample affects exactly one teacher's training, bounding the privacy impact.

## Training the Teacher Ensemble

With partitions defined, we train all 250 teachers on their respective data subsets. Each teacher uses the `MLP` class from `htb_ai_library`, a multilayer perceptron with hidden layers `[128, 64]`. This architecture provides sufficient capacity for the 10-class digit classification task while remaining efficient to train 250 times. The same architecture will be used for the student model, ensuring capacity matches between teacher knowledge and student learning.

First, we initialize the container and create a shared holdout loader for consistent evaluation:

Code: python

```python
teachers = []
holdout_loader = create_dataloader(X_holdout_norm, y_holdout, 128, shuffle=False)
```

Using the same `holdout_loader` across all teachers ensures we measure generalization consistently. Now we iterate through each partition, creating and training a fresh model:

Code: python

```python
for i in tqdm(range(num_teachers), desc="Training teachers"):
    partition_idx = teacher_partitions[i]
    X_teacher = X_private_norm[partition_idx]
    y_teacher = y_private[partition_idx]

    teacher = MLP(
        input_size=num_features,
        hidden_layers=TEACHER_CONFIG['hidden_layers'],
        num_classes=DATASET_CONFIG['num_classes'],
        dropout=TEACHER_CONFIG['dropout']
    )

    train_loader = create_dataloader(X_teacher, y_teacher, TEACHER_CONFIG['batch_size'])
    train_model(teacher, train_loader, holdout_loader, device=DEVICE,
                epochs=TEACHER_CONFIG['epochs'], learning_rate=TEACHER_CONFIG['learning_rate'])
    teachers.append(teacher)

print(f"\nTeacher ensemble trained: {len(teachers)} teachers")
```

Fresh models for each teacher (rather than shared weights) maintain the privacy guarantee that each teacher knows only its partition. Weight sharing would create dependencies between teachers, undermining this isolation. Individual teacher accuracy typically ranges from 74-82% on the holdout set. This variance reflects both the randomness of small training sets and the inherent difficulty of certain digit pairs (3 vs 8, 4 vs 9).

## Ensemble Voting

With all teachers trained, we implement the voting mechanism that aggregates their predictions:

Code: python

```python
def get_teacher_votes(teachers, X, device):
    """
    Get vote counts from all teachers.

    Returns:
        np.ndarray: Vote counts of shape (num_samples, num_classes).
    """
    num_samples = X.shape[0]
    num_classes = DATASET_CONFIG['num_classes']
    votes = np.zeros((num_samples, num_classes), dtype=np.int32)

    for teacher in teachers:
        preds = get_model_predictions(teacher, X, device)
        predictions = np.argmax(preds, axis=1)

        for i, pred in enumerate(predictions):
            votes[i, pred] += 1

    return votes
```

We iterate through all teachers, obtaining each one’s predictions on the input samples. Each teacher produces logits, and `argmax` converts these to class predictions. We increment the vote count for the predicted class, building a histogram of votes for each sample. The output array has shape `(num_samples, num_classes)` where `votes[i, j]` counts how many teachers predicted class j for sample i.

