
`Data transformations` improve the representation and distribution of features, making them more suitable for machine learning models. These transformations ensure that models can efficiently capture underlying patterns by converting categorical variables into machine-readable formats and addressing skewed numerical distributions. They also enhance trained models' stability, interpretability, and predictive performance.

## Encoding Categorical Features

`Encoding` converts categorical values into numeric form so machine learning algorithms can utilize these features. Depending on the situation, you can choose:

- `OneHotEncoder` for binary indicator features that represent each category separately.
- `LabelEncoder` for integer codes, though this may imply unintended order.
- `HashingEncoder` or frequency-based methods to handle high-cardinality features and control feature space size.

After encoding, verify that the transformed features are meaningful and do not introduce artificial ordering.

### One-Hot Encoding

`One-hot encoding` takes a categorical feature and converts it into a set of new binary features, where each binary feature corresponds to one possible category value. This process creates a set of indicator columns that hold `1` or `0`, indicating the presence or absence of a particular category in each row.

For example, consider the categorical feature `color`, which can take on the values `red`, `green`, or `blue`. In a dataset, you might have rows where `color` is `red` in one instance, `green` in another, and so on. By applying `one-hot encoding`, instead of keeping a single column with values like `red`, `green`, or `blue`, the encoding creates three new binary columns:

- `color_red`
- `color_green`
- `color_blue`

Each of these new columns corresponds to one of the original categories. If a row had `color` set to `red`, the `color_red` column for that row would be `1`, and the other two columns (`color_green` and `color_blue`) would be `0`. Similarly, if `color` was originally `green`, then the `color_green` column would be `1`, while the `color_red` and `color_blue` columns would be `0`.

![Table showing one-hot encoding of colors: red, green, blue.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/data_encoding.png)

This approach prevents models from misinterpreting category values as numeric hierarchies. However, it can increase the number of features if a category has many unique values.

In this case, we are going to encode the `protocol` feature.

Code: python

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['protocol']])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['protocol']))
df = pd.concat([df.drop('protocol', axis=1), encoded_df], axis=1)
```

The original `protocol` feature is replaced with distinct binary columns, ensuring the model interprets each category independently.

## Handling Skewed Data

When a feature is `skewed`, its values are unevenly distributed, often with most observations clustered near one end and a few extreme values stretching out the distribution. Such skew can affect the performance of machine learning models, especially those sensitive to outliers or that assume more uniform or normal-like data distributions.

`Scaling` or transforming these skewed features helps models better capture patterns in the data. One common transformation is applying a `log` transform to compress large values more than small ones, resulting in a more balanced distribution and less dominated by outliers. By doing this, models often gain improved stability, accuracy, and generalization ability.

Below, we show how to apply a `log` transform using the `log1p` function. This approach adds 1 to each value before taking the `log`, ensuring that the transform is defined even for values at or near zero.

Code: python

```python
import numpy as np

# Apply logarithmic transformation to a skewed feature to reduce its skewness
df["bytes_transferred"] = np.log1p(df["bytes_transferred"])  # Add 1 to avoid log(0)
```

The code above transforms the `bytes_transferred` feature. Before this transformation, the feature might have had a few very large values, overshadowing the majority of smaller observations. After the transformation, the distribution is evener, helping the model treat all data points fairly and reducing the risk of overfitting outliers.

![Two histograms: original distribution of bytes transferred and log-transformed distribution.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/log_histogram.png)

Visual comparisons of the distribution before and after the transform (as shown by the above figure) confirm that the original skew has been substantially reduced. Although no information is lost, the model now views the data through a lens that downplays extreme cases and highlights underlying patterns more clearly.

## Data Splitting

`Data splitting` involves dividing a dataset into three distinct subsets—`training`, `validation`, and `testing`—to ensure reliable model evaluation. By having separate sets, you can train your model on one subset, fine-tune it on another, and finally test its performance on data it has never seen before.

- `Training Set`: Used to fit the model. Typically accounts for around 60-80% of the entire dataset.
- `Validation Set`: Used for tuning hyperparameters and model selection. Often around 10-20% of the entire dataset.
- `Test Set`: Used only after all model selections and tuning are complete. Often around 10-20% of the entire dataset.

The code below demonstrates one approach using `train_test_split` from `scikit-learn`. The initial split allocates 80% of the data for training and 20% for testing. A subsequent split divides the 80% training portion into 60% for final training and 20% for validation.

Note that `test_size=0.25` in the second split refers to 25% of the previously created training subset (which is 80% of the data). In other words, `0.8 × 0.25 = 0.2` (20% of the entire dataset), leaving 60% for training and 20% for validation overall.

Code: python

```python
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop("threat_level", axis=1)
y = df["threat_level"]

# Initial split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# Second split: from the 80% training portion, allocate 60% for final training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
```

These subsets support a structured workflow:

- Train the model on `X_train` and `y_train`.
- Tune hyperparameters or compare different models using `X_val` and `y_val`.
- Finally, evaluate the performance on the untouched `X_test` and `y_test`.

