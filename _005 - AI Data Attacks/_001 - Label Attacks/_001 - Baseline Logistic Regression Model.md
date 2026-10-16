

Before executing the attack, we need to establish `baseline performance` so we have something to compare the effects of the poisoned model with. We will train a `Logistic Regression` model on the original, clean training data (`X_train`, `y_train`). This baseline represents the model's expected behavior and accuracy under normal, `non-adversarial conditions`.

As outlined in the "[Fundamentals of AI](https://academy.hackthebox.com/module/details/290)" module, `Logistic Regression` is fundamentally a classification algorithm used for predicting binary outcomes.

For a given review represented by its feature vector 𝐱i=(xi1,xi2), the model first calculates a linear combination zi using weights 𝐰=(w1,w2) and a bias term b:

zi=𝐰T𝐱i+b=w1xi1+w2xi2+b

This value zi represents the `log-odds` (or logit) of the review having positive sentiment (yi=1). It quantifies the linear relationship between the derived features and the log-odds of a positive classification.

zi=log⁡(P(yi=1|𝐱i)1−P(yi=1|𝐱i))

To convert the log-odds zi into a probability pi=P(yi=1|𝐱i) (the probability of the review being `positive`), the model applies the `sigmoid function`, σ:

pi=σ(zi)=11+e−zi=11+e−(𝐰T𝐱i+b)

The sigmoid function squashes the output zi into the range [0,1], representing the model’s estimated probability that the review 𝐱i has `positive` sentiment.

During training, the model learns the optimal parameters 𝐰 and b by minimizing a `loss function` over the training set (`X_train`, `y_train`). The goal is to find parameters that make the predicted probabilities pi as close as possible to the true sentiment labels yi. The standard loss function for binary classification is the `binary cross-entropy` or `log-loss`:

L(𝐰,b)=−1N∑i=1N[yilog(pi)+(1−yi)log(1−pi)]

Here, N is the number of training reviews, yi is the true sentiment label (0 or 1) for the i-th review, and pi is the model’s predicted probability of positive sentiment for that review. Optimization algorithms like `gradient descent` iteratively adjust 𝐰 and b to minimize this loss L.

Once trained, the model uses the learned 𝐰 and b to predict the sentiment of new, unseen reviews. For a new review 𝐱, it calculates the probability p=σ(𝐰T𝐱+b). Typically, if p≥0.5, the review is classified as `positive` (Class 1); otherwise, it’s classified as `negative` (Class 0).

The `decision boundary` is the line (in our 2D feature space) where the model is exactly uncertain (p=0.5), which occurs when z=𝐰T𝐱+b=0. This linear boundary separates the feature space into regions predicted as `negative` and `positive` sentiment. The training process finds the line that best separates the clusters in the training data.

We now train this baseline model on our clean data and evaluate its accuracy on the unseen test set.

Code: python

```python
# Initialize and train the Logistic Regression model
baseline_model = LogisticRegression(random_state=SEED)
baseline_model.fit(X_train, y_train)

# Predict on the test set
y_pred_baseline = baseline_model.predict(X_test)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")


# Prepare to plot the decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary of a trained classifier on a 2D dataset.

    Parameters:
    - model: The trained classifier object (must have a .predict method).
    - X (np.ndarray): Feature data (n_samples, 2).
    - y (np.ndarray): Labels (n_samples,).
    - title (str): The title for the plot.
    """
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 6))
    # Plot the decision boundary contour
    plt.contourf(
        xx, yy, Z, cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]), alpha=0.3
    )

    # Plot the data points
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=plt.cm.colors.ListedColormap([azure, nugget_yellow]),
        edgecolors=node_black,
        s=50,
        alpha=0.8,
    )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)

    # Create a legend manually
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Negative Sentiment (Class 0)",
            markersize=10,
            markerfacecolor=azure,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Positive Sentiment (Class 1)",
            markersize=10,
            markerfacecolor=nugget_yellow,
        ),
    ]
    plt.legend(handles=handles, title="Classes")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


# Plot the decision boundary for the baseline model
plot_decision_boundary(
    baseline_model,
    X_train,
    y_train,
    title=f"Baseline Model Decision Boundary\nAccuracy: {baseline_accuracy:.4f}",
)
```

The resulting plot shows the linear decision boundary learned by the baseline model, effectively separating the simulated `Negative` and `Positive` sentiment clusters in the training data. The high accuracy score indicates it generalizes well to the unseen test data.

![Scatter plot titled 'Baseline Model Decision Boundary' with accuracy 0.9933, showing Negative Sentiment (Class 0, blue) and Positive Sentiment (Class 1, orange) across Feature 1 and Feature 2.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/302/label_flipping_baseline.png)

