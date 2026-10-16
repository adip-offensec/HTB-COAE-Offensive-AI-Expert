
# Training the Robust Model

With the core components in place, this section implements the adversarial training loop that transforms a vulnerable classifier into a robust one. You will generate evaluation examples across multiple epsilon values, implement the training loop with epsilon spread sampling, and execute the complete training pipeline.

---

## Generating Evaluation Examples

The evaluator requires pre-generated adversarial examples stored in `adv_examples.safetensors`. This ensures consistent evaluation across different training runs. To generate examples at all epsilon values in the spread, we first collect clean samples:

Code: python

```python
def generate_adversarial_examples(model, test_loader, device, num_samples=500):
    """Generate adversarial examples across multiple epsilon values."""
    model.eval()

    # Collect clean samples
    clean_images_list = []
    clean_labels_list = []
    collected = 0

    print(f"Collecting {num_samples} clean samples...")
    for images, labels in test_loader:
        if collected >= num_samples:
            break
        batch_size = min(images.size(0), num_samples - collected)
        clean_images_list.append(images[:batch_size])
        clean_labels_list.append(labels[:batch_size])
        collected += batch_size

    clean_images = torch.cat(clean_images_list, dim=0)
    clean_labels = torch.cat(clean_labels_list, dim=0)
```

We iterate through the test loader, collecting exactly `num_samples` images and their labels. The `min()` call handles the final batch, which may have fewer samples than needed. After collection, we concatenate all tensors into single arrays for efficient batch processing.

Code: python

```python
    # Generate adversarial examples at each epsilon
    fgsm_by_epsilon = {}
    ifgsm_by_epsilon = {}

    print(f"\nGenerating adversarial examples across epsilon spread: {EPSILON_SPREAD}")

    for eps in EPSILON_SPREAD:
        print(f"\n  Generating at epsilon={eps}...")
        fgsm_images_list = []
        ifgsm_images_list = []

        batch_size = 128
        pbar = tqdm(total=num_samples, desc=f"  eps={eps}")
```

We create dictionaries to store adversarial examples keyed by epsilon value. For each epsilon in the spread (0.1 through 1.0), we generate both FGSM and I-FGSM variants in batches of 128 for memory efficiency.

Code: python

```python
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            images = clean_images[i:end_idx].to(device)
            labels = clean_labels[i:end_idx].to(device)

            model.train()  # Need train mode for gradient computation
            fgsm_images = fgsm_attack(model, images, labels, eps)
            ifgsm_images = i_fgsm_attack(model, images, labels, eps)
            model.eval()

            fgsm_images_list.append(fgsm_images.cpu())
            ifgsm_images_list.append(ifgsm_images.cpu())
            pbar.update(end_idx - i)

        pbar.close()

        fgsm_by_epsilon[eps] = torch.cat(fgsm_images_list, dim=0)
        ifgsm_by_epsilon[eps] = torch.cat(ifgsm_images_list, dim=0)
```

Each batch moves to the device, generates both attack types, then moves results back to CPU to free GPU memory. After processing all batches for a given epsilon, we concatenate into single tensors and store in our dictionaries.

Code: python

```python
    return {
        'clean_images': clean_images,
        'clean_labels': clean_labels,
        'fgsm_images': fgsm_by_epsilon[EPSILON],
        'ifgsm_images': ifgsm_by_epsilon[EPSILON],
        'epsilon': EPSILON,
        'epsilon_spread': EPSILON_SPREAD,
        'fgsm_by_epsilon': fgsm_by_epsilon,
        'ifgsm_by_epsilon': ifgsm_by_epsilon
    }
```

We return a dictionary containing clean samples, the full epsilon spread dictionaries, and the primary epsilon (0.3) examples at the top level for backward compatibility. Generating examples from the baseline model ensures attacks are calibrated against a vulnerable target, representing realistic attack conditions.

---

## Implementing the Training Loop

Now we build the complete adversarial training function, broken down block by block. The function signature and setup establish the optimizer, scheduler, and loss function:

Code: python

```python
def train_adversarial(model, train_loader, test_loader, device,
                      epochs=25, lr=0.001, epsilon=EPSILON):
    """Train model with adversarial training using epsilon spread."""
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
```

AdamW includes decoupled weight decay for better generalization. We use cosine annealing for learning rate scheduling, as discussed in the previous section.

The outer epoch loop initializes tracking variables and wraps the DataLoader with tqdm for progress display:

Code: python

```python
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
```

The inner batch loop implements the core adversarial training steps. First, sample a random epsilon from the spread and generate adversarial examples at that intensity:

Code: python

```python
            batch_epsilon = np.random.choice(EPSILON_SPREAD)
            adv_images = fgsm_attack(model, images, labels, batch_epsilon)
```

This epsilon spread sampling implements the technique described in the previous section.

Next, combine clean and adversarial images into a single batch and shuffle to prevent position-based pattern learning:

Code: python

```python
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            perm = torch.randperm(combined_images.size(0))
            combined_images = combined_images[perm]
            combined_labels = combined_labels[perm]
```

We shuffle to prevent position-based pattern learning, as clean examples would otherwise always appear first.

Perform the forward pass, compute loss, backpropagate with gradient clipping, and update weights:

Code: python

```python
            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

Gradient clipping (discussed in the theory section) stabilizes early training.

Track metrics and update the progress bar:

Code: python

```python
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += combined_labels.size(0)
            correct += predicted.eq(combined_labels).sum().item()

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
```

At the end of each epoch, step the learning rate scheduler and periodically evaluate both clean and robust accuracy:

Code: python

```python
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            clean_acc = evaluate_accuracy(model, test_loader, device)
            adv_acc = evaluate_adversarial_accuracy(
                model, test_loader, device, epsilon, num_batches=20
            )
            print(f"\n  Epoch {epoch+1}: Clean={clean_acc:.1f}%, Robust={adv_acc:.1f}%")

    return model
```

Evaluating every 5 epochs provides feedback on training progress without excessive overhead.

---

## Running the Training

With all functions defined, we can now execute the training pipeline. The complete workflow trains a baseline model, generates evaluation examples, then performs adversarial training.

Start by setting up the environment and loading data:

Code: python

```python
set_reproducibility(1337)
device = get_device()
print(f"Using device: {device}")

train_loader, test_loader = get_mnist_loaders(normalize=True)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
```

  Training the Robust Model

```shell-session
Using device: cuda
Train batches: 469, Test batches: 79
```

#### Training the Baseline Model

Using the `train_model` function from the previous section, train the baseline model and evaluate its robustness:

Code: python

```python
baseline_model = LeNet5()
baseline_model = train_model(baseline_model, train_loader, test_loader,
                              device=device, epochs=10, learning_rate=0.001)

adv_acc = evaluate_adversarial_accuracy(baseline_model, test_loader, device, EPSILON)
print(f"Baseline Model - Robust: {adv_acc:.1f}%")
save_file(baseline_model.state_dict(), "baseline_model.safetensors")
```

The baseline achieves ~99% clean accuracy but only ~73% robust accuracy, confirming its vulnerability.

#### Generating Adversarial Examples

With the baseline trained, we generate FGSM and I-FGSM adversarial examples for consistent evaluation:

Code: python

```python
adv_data = generate_adversarial_examples(
    baseline_model, test_loader, device, num_samples=500
)

save_adversarial_examples(adv_data, "adv_examples.safetensors")
print("Adversarial examples saved to adv_examples.safetensors")
```

This creates `adv_examples.safetensors` containing adversarial examples at all epsilon values. The evaluator uses this file for consistent testing across different training runs.

#### Running Adversarial Training

Now we train a new model with adversarial training. We initialize a fresh model and evaluate its starting performance:

Code: python

```python
model = LeNet5()
model.to(device)

clean_acc = evaluate_accuracy(model, test_loader, device)
adv_acc = evaluate_adversarial_accuracy(model, test_loader, device, EPSILON)
print(f"Before training - Clean: {clean_acc:.1f}%, Robust: {adv_acc:.1f}%")
```

  Training the Robust Model

```shell-session
Before training - Clean: 11.3%, Robust: 1.9%
```

A randomly initialized model performs at chance level (approximately 10% for 10 classes). Now run the adversarial training:

Code: python

```python
model = train_adversarial(model, train_loader, test_loader, device, epochs=10)
```

Training progress shows gradual improvement as the model learns to handle adversarial perturbations:

  Training the Robust Model

```shell-session
Epoch 5: Clean=98.8%, Robust=93.5%
Epoch 10: Clean=99.0%, Robust=95.4%
```

The pattern to watch for is robust accuracy crossing 92% while clean accuracy stays above 94%. After training completes, run a final evaluation on the full test set:

Code: python

```python
clean_acc = evaluate_accuracy(model, test_loader, device)
adv_acc = evaluate_adversarial_accuracy(model, test_loader, device, EPSILON)
print(f"Final - Clean: {clean_acc:.1f}%, Robust: {adv_acc:.1f}%")
```

  Training the Robust Model

```shell-session
Final - Clean: 99.0%, Robust: 95.4%
```

Save the trained model for evaluation:

Code: python

```python
save_file(model.state_dict(), "robust_model.safetensors")
print("Model saved to robust_model.safetensors")
```

---

## Evaluator Verification

To verify your model meets the success criteria, run the provided evaluator against your saved model. You can download the evaluator [here](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/322/adversarial_training.zip).

  Training the Robust Model

```shell-session
@htb[/htb]$ python evaluate_robustness.py --model-path robust_model.safetensors
```

The evaluator tests against pre-generated adversarial examples at multiple epsilon values, ensuring consistent evaluation across different runs. The next section explains the evaluator output and how to interpret the robustness metrics.

