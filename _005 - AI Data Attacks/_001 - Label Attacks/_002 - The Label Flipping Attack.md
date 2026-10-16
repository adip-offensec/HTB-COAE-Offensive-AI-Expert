

With an established baseline, we can now execute the actual attack, and to do this, we will create a function that will take the original training labels (`y_train`, representing the true sentiments) and a `poisoning percentage` as input. It will randomly select the specified fraction of training data points (reviews) and flip their labels - changing `Negative` (0) to `Positive` (1) and `Positive` (1) to `Negative` (0).

The implication of this is significant. As we have established, the model learns its parameters (𝐰,b) by minimizing the average `log-loss`, L, across the training dataset, the whole point of training is to find the 𝐰 and b that make this loss L as small as possible, meaning the predicted probabilities pi align well with the true labels yi.

When we flip a label for a specific instance from its true value yi to an incorrect value yi′, we directly corrupt the contribution of that instance to the overall loss calculation. For example, consider an instance 𝐱i that truly belongs to class 0 (so yi=0) but its label is flipped to yi′=1. The term for this instance inside the sum changes from −[0⋅log⁡(pi)+(1−0)log⁡(1−pi)]=−log⁡(1−pi) to −[1⋅log⁡(pi)+(1−1)log⁡(1−pi)]=−log⁡(pi).

If the model, based on the features 𝐱i, correctly learns to predict a low probability pi for class 1 (since the instance truly belongs to class 0), the original term −log⁡(1−pi) would be small, but the corrupted term −log⁡(pi) becomes very large as pi→0.

This large error signal for the flipped instance strongly influences the optimization process. It forces the algorithm to adjust the parameters 𝐰 and b not only fit the correctly labeled data, but also to try and accommodate these poisoned points, and in doing so, it pushes the learned `decision boundary` defined by 𝐰T𝐱+b=0, `away from the optimal position` determined by the true underlying data distribution.

## flip_labels

To execute this attack, we will implement a function to contain all logic: `flip_labels`. This function takes the original training labels (`y_train`) and a `poison_percentage` as input, specifying the fraction of labels to flip.

First, we define the function signature and ensure the provided `poison_percentage` is a valid value between 0 and 1. This prevents nonsensical inputs. We also calculate the absolute number of labels to flip (`n_to_flip`) based on the total number of samples and the specified percentage.

Code: python

```python
def flip_labels(y, poison_percentage):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")

    n_samples = len(y)
    n_to_flip = int(n_samples * poison_percentage)

    if n_to_flip == 0:
        print("Warning: Poison percentage is 0 or too low to flip any labels.")
        # Return unchanged labels and empty indices if no flips are needed
        return y.copy(), np.array([], dtype=int)
```

Next, we select which specific reviews (data points) will have their sentiment labels flipped. We use a NumPy random number generator (`rng_instance`) seeded with our global `SEED` (or the function's seed parameter) for reproducible random selection. The `choice` method selects `n_to_flip` unique indices from the range `0` to `n_samples - 1` without replacement. These `flipped_indices` identify the exact reviews targeted by the attack.

Code: python

```python
    # Use the defined SEED for the random number generator
    rng_instance = np.random.default_rng(SEED)
    # Select unique indices to flip
    flipped_indices = rng_instance.choice(n_samples, size=n_to_flip, replace=False)
```

Now, we perform the actual label flipping. We create a copy of the original label array (`y_poisoned = y.copy()`) to avoid altering the original data. For the elements at the `flipped_indices`, we invert their labels: `0` becomes `1`, and `1` becomes `0`. A concise way to do this is `1 - label` for binary 0/1 labels, or using `np.where` for clarity.

Code: python

```python
    # Create a copy to avoid modifying the original array
    y_poisoned = y.copy()

    # Get the original labels at the indices we are about to flip
    original_labels_at_flipped = y_poisoned[flipped_indices]

    # Apply the flip: if original was 0, set to 1; otherwise (if 1), set to 0
    y_poisoned[flipped_indices] = np.where(original_labels_at_flipped == 0, 1, 0)

    print(f"Flipping {n_to_flip} labels ({poison_percentage * 100:.1f}%).")
```

Lastly, the function returns the `y_poisoned` array containing the corrupted labels and the `flipped_indices` array, allowing us to track which reviews were affected.

Code: python

```python
    return y_poisoned, flipped_indices
```

We also need a function to plot the data so its easy to see the effects of the attack.

Code: python

```python
def plot_poisoned_data(
    X,
    y_original,
    y_poisoned,
    flipped_indices,
    title="Poisoned Data Visualization",
    target_class_info=None,
):
    """
    Plots a 2D dataset, highlighting points whose labels were flipped.

    Parameters:
    - X (np.ndarray): Feature data (n_samples, 2).
    - y_original (np.ndarray): The original labels before flipping (used for context if needed, currently unused in logic but good practice).
    - y_poisoned (np.ndarray): Labels after flipping.
    - flipped_indices (np.ndarray): Indices of the samples that were flipped.
    - title (str): The title for the plot.
    - target_class_info (int, optional): The class label of the points that were targeted for flipping. Defaults to None.
    """
    plt.figure(figsize=(12, 7))

    # Identify non-flipped points
    mask_not_flipped = np.ones(len(y_poisoned), dtype=bool)
    mask_not_flipped[flipped_indices] = False

    # Plot non-flipped points (color by their poisoned label, which is same as original)
    plt.scatter(
        X[mask_not_flipped, 0],
        X[mask_not_flipped, 1],
        c=y_poisoned[mask_not_flipped],
        cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
        edgecolors=node_black,
        s=50,
        alpha=0.6,
        label="Unchanged Label",  # Keep this generic
    )

    # Determine the label for flipped points in the legend
    if target_class_info is not None:
        flipped_legend_label = f"Flipped (Orig Class {target_class_info})"
        # You could potentially use target_class_info to adjust facecolor if needed,
        # but current logic colors by the new label which is often clearer.
    else:
        flipped_legend_label = "Flipped Label"

    # Plot flipped points with a distinct marker and outline
    if len(flipped_indices) > 0:
        # Color flipped points according to their new (poisoned) label
        plt.scatter(
            X[flipped_indices, 0],
            X[flipped_indices, 1],
            c=y_poisoned[flipped_indices],  # Color by the new label
            cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
            edgecolors=malware_red,  # Highlight edge in red
            linewidths=1.5,
            marker="X",  # Use 'X' marker
            s=100,
            alpha=0.9,
            label=flipped_legend_label,  # Use the determined label
        )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)

    # Create legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 0 Point (Azure)",
            markersize=10,
            markerfacecolor=azure,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 1 Point (Yellow)",
            markersize=10,
            markerfacecolor=nugget_yellow,
            linestyle="None",
        ),
        # Add the flipped legend entry using the label
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            label=flipped_legend_label,
            markersize=12,
            markeredgecolor=malware_red,
            markerfacecolor=hacker_grey,
            linestyle="None",
        ),
    ]
    plt.legend(handles=handles, title="Data Points")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.show()
```

