

# Private Aggregation of Teacher Ensembles

Earlier sections demonstrated how membership inference attacks exploit overfitting to reveal which samples trained a model, and how DP-SGD provides mathematical privacy guarantees by injecting noise during gradient computation. Here we explore `PATE` (Private Aggregation of Teacher Ensembles), a distinct approach that achieves privacy through architectural separation instead of training-time perturbation.

PATE takes a counterintuitive approach to privacy. Instead of protecting the model during training, we ensure the deployed model never accesses sensitive data at all. The idea is straightforward: train multiple `teacher models` on disjoint partitions of sensitive data, then use their aggregated and noisy predictions to label `public unlabeled data`, on which a `student model` trains. The student never directly sees the sensitive data, yet learns to make accurate predictions.

This technique builds on the same teacher–student and knowledge distillation ideas that appear in many modern ML systems, and has been explored extensively in academic and industrial research on privacy-preserving learning rather than being tied to any single deployed system.

## The Core Privacy Mechanism

Traditional machine learning creates a direct connection between model parameters and individual training samples, producing the memorization patterns that membership inference exploits. PATE breaks this connection through knowledge distillation with noise injection. The approach relies on `teacher-student transfer`, where multiple teachers trained on disjoint data partitions dilute any single sample's influence on the final model. Privacy emerges from `aggregation with noise`: we add calibrated Laplacian noise to vote counts before selecting majority predictions. The result is an `information bottleneck` that compresses megabytes of private data into kilobytes of noisy labels, fundamentally limiting what attackers can learn.

#### Teacher-Student Knowledge Transfer

We operate PATE in two distinct phases. In the first phase, we partition the sensitive training data into n disjoint subsets and train one teacher model on each partition. Because each teacher sees only a fraction of the total data, any single teacher has limited knowledge about the complete dataset. The privacy benefit comes from aggregation: to learn about a specific training sample, an attacker would need to identify which teacher was trained on that sample and extract information from that particular teacher. This becomes increasingly difficult as the number of teachers grows.

In the second phase, we use the teacher ensemble to generate `pseudo-labels` for a separate set of unlabeled public data. For each public sample, all teachers vote on the predicted class. We add calibrated noise to this vote count before selecting the majority class as the label. The student model then trains only on this noisy-labeled public data, never accessing the original sensitive samples.

The mathematical formulation for the aggregation mechanism captures this process precisely:

ŷ=arg⁡max⁡j(nj(x)+Lap(1ϵ))

Here nj(x) represents the number of teachers voting for class j on input x, and we add independent Laplacian noise with scale 1ϵ to each vote count before selecting the class with the highest noisy count. The privacy parameter ϵ controls the noise magnitude: smaller values provide stronger privacy but potentially noisier labels. With 250 teachers and ϵ=0.10 per query (noise scale 20), strong consensus (200+ votes) survives noise while weak consensus may flip, providing both useful labels and meaningful privacy.

#### Why Aggregation Provides Privacy

Consider what happens when a teacher votes on a sample from its own training set versus a sample it has never seen. Each individual teacher may exhibit the overfitting behavior exploited in membership inference attacks. However, when we aggregate votes across many teachers, only one of them (the one trained on that particular sample's partition) has this memorized knowledge. The other teachers vote based on genuine generalization, diluting the membership signal. Adding noise further obscures whether the consensus came from memorization or legitimate pattern recognition.

The privacy guarantee strengthens with both the number of teachers and the degree of consensus. If all teachers agree on a prediction, the noise is unlikely to flip such a strong consensus, and the resulting label reveals minimal information because the same prediction would occur regardless of any single training sample. Conversely, if teachers split nearly evenly, the noise dominates and the resulting label carries little information about any individual training sample.

## PATE Architecture Overview

The PATE pipeline processes data through distinct stages, each contributing to the privacy guarantee. Data flows from partitioned private data through teacher voting to noisy aggregation to student training. The information bottleneck compresses high-dimensional private data into low-dimensional noisy labels. Privacy budget composition tracks cumulative expenditure across labeling queries using the moments accountant.

#### Data Flow

We begin with two datasets: `sensitive private data` Dprivate containing samples we must protect, and `public unlabeled data` Dpublic that requires no protection but lacks labels. We partition the sensitive data into n disjoint subsets D1,D2,…,Dn, with each subset used to train a corresponding teacher model T1,T2,…,Tn.

When labeling public data, each sample x∈Dpublic passes to all teachers simultaneously. Each teacher produces a class prediction, effectively casting a vote. We aggregate these votes into a histogram over classes, add noise to the histogram, and select the class with the highest noisy count as the pseudo-label ỹ for that sample. The student model S trains on pairs (x,ỹ) using standard supervised learning.

![PATE architecture diagram showing the privacy-preserving pipeline. Sensitive private data is partitioned into n disjoint subsets, each training a separate teacher model. Public unlabeled data flows through all teachers, whose votes are aggregated, noise is added via Laplacian mechanism, and pseudo-labels are generated to train the final student model.](https://cdn.services-k8s.prod.aws.htb.systems/content/sections/335_pate_diagram.png)

#### The Information Bottleneck

We can interpret PATE's privacy guarantee intuitively through information theory. The student model receives only noisy class labels, a small number of bits per query. From a dataset of tens of thousands of private samples consuming megabytes of storage, the student receives only thousands of label values total. This extreme compression, combined with the formal differential privacy guarantee, severely limits how much information an attacker can recover about any individual record.

Consider a concrete example with the `MNIST` dataset: 48,000 private handwritten digit images with 784 features each occupy approximately 150 MB when stored as 32-bit floats. Through PATE with 5,000 queries, the student receives 5,000 class labels (each an integer 0-9, requiring about 4 bits). That is roughly 2.5 KB of information. The compression ratio exceeds 60,000:1. No matter how sophisticated the attack, recovering megabytes of private information from kilobytes of noisy labels violates fundamental information-theoretic limits.

#### Privacy Budget Composition

Each time the teacher ensemble labels a public sample, it expends some privacy budget. Total privacy expenditure depends on the number of queries and the noise level at each query.

For m labeling queries, each with privacy parameter ϵ0, the total privacy cost is not simply m⋅ϵ0. This naive composition bound would give ϵ=500 for 5,000 queries at ϵ0=0.10, representing negligible privacy. Fortunately, tighter bounds exist. The `advanced composition theorem` provides a bound proportional to m⋅ϵ0 instead of m⋅ϵ0. This sublinear scaling makes PATE practical for labeling datasets of reasonable size.

In practice, we use the `moments accountant` technique to compute these tighter bounds. This approach tracks the logarithm of the moment generating function of the privacy loss random variable across queries, exploiting the statistical properties of independent noise additions to produce bounds even tighter than the basic advanced composition theorem. With 5,000 queries at noise scale 20.0, the moments accountant yields approximately ϵ=8.81, a dramatic improvement over naive composition’s ϵ=500.

The formal privacy guarantee states that the student model satisfies (ϵ,δ)-differential privacy with respect to the sensitive training data, where ϵ depends on the number of teachers, noise scale, and number of labeling queries. Once trained, the student can be queried unlimited times without additional privacy cost because it never accesses the sensitive data directly.

## Implementation Approach

We implement PATE on the `MNIST` dataset, the standard benchmark used in the original PATE papers by Papernot et al. `MNIST` contains 60,000 handwritten digit images with 784 features (28x28 pixels), and the task is to predict one of 10 digit classes (0-9). This dataset is ideal for demonstrating PATE because it allows strong teacher consensus across the 250-teacher ensemble, enabling meaningful privacy guarantees while maintaining high accuracy.

#### Data Organization

To properly evaluate PATE, we need three distinct data pools plus a holdout set for evaluation. The `private data` trains the teacher ensemble exclusively and represents the sensitive images we must protect. The `public data` provides unlabeled samples that teachers label for student training, simulating the common scenario where unlabeled data is freely available but labeled data requires accessing sensitive sources. The `holdout data` remains untouched by both teachers and student, reserved solely for final model evaluation.

This three-way separation ensures the student never accesses private data while maintaining clean evaluation on genuinely held-out samples.

## Environment Setup

To begin, we establish the computational environment with necessary libraries:

Code: python

```python
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
from tqdm import tqdm

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    MLP, get_mnist_loaders,
    create_dataloader, train_model, evaluate_accuracy, get_model_predictions,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, MALWARE_RED, AQUAMARINE, NUGGET_YELLOW,
)
```

We import standard scientific computing libraries alongside the `htb_ai_library` module, which provides the `MLP` architecture, data loading utilities, and training functions. Now we display a banner confirming the environment is ready:

Code: python

```python
print("=" * 60)
print("PATE: PRIVATE AGGREGATION OF TEACHER ENSEMBLES")
print("Teacher-Student Privacy Through Knowledge Distillation")
print("=" * 60)
```

#### Configuration Constants

These configuration parameters control the PATE pipeline and balance privacy protection against model utility. We set these values to match the original PATE papers and demonstrate meaningful privacy guarantees. First, we establish reproducibility and detect available hardware:

Code: python

```python
RANDOM_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_reproducibility(RANDOM_SEED)
use_htb_style()

PLOT_CONFIG = {"figsize": (10, 6), "dpi": 150}
DATASET_CONFIG = {"name": "mnist", "num_classes": 10, "num_features": 784}
```

The seed value `1337` ensures identical results across runs, while `DEVICE` automatically selects GPU acceleration when available. Now we configure the teacher ensemble, the heart of PATE's privacy mechanism:

Code: python

```python
TEACHER_CONFIG = {
    "num_teachers": 250,  # Original paper configuration
    "hidden_layers": [128, 64],
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
}
```

Why 250 teachers? This matches the original PATE papers and provides exceptionally strong aggregation. With 48,000 private samples, each teacher receives approximately 192 samples. Despite this limited data, individual teachers achieve 74-82% accuracy on `MNIST` because digit recognition is relatively straightforward. The `[128, 64]` hidden layer configuration provides sufficient capacity for the 10-class task while remaining efficient to train 250 times.

Code: python

```python
AGGREGATION_CONFIG = {
    "noise_scale": 20.0,  # Strong privacy with high consensus
    "num_student_queries": 5000,
}

STUDENT_CONFIG = {
    "hidden_layers": [128, 64],
    "dropout": 0.2,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
}
```

The noise scale of 20.0 yields per-query privacy of ϵ0=220=0.10, where the numerator 2 comes from voting sensitivity (adding one sample can shift one vote from one class to another, changing two vote counts by 1 each). With 250 teachers achieving strong consensus (typically 200+ votes for the winning class), this noise level preserves approximately 87% label accuracy while providing strong privacy. Over 5,000 queries, advanced composition gives total privacy of approximately ϵ=8.81.

