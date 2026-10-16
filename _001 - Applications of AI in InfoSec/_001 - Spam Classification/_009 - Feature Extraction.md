

`Feature extraction` transforms preprocessed SMS messages into numerical vectors suitable for machine learning algorithms. Since models `cannot directly process raw text data`, they rely on numeric representations—such as counts or frequencies of words—to identify patterns that differentiate spam from ham.

## Representing Text as Numerical Features

A common way to represent text numerically is through a `bag-of-words` model. This technique constructs a vocabulary of unique terms from the dataset and represents each message as a vector of term counts. Each element in the vector corresponds to a term in the vocabulary, and its value indicates how often that term appears in the message.

Using only `unigrams` (individual words) does not preserve the original word order; it treats each document as a collection of terms and their frequencies, independent of sequence.

To introduce a limited sense of order, we also include `bigrams`, which are pairs of consecutive words. By incorporating bigrams, we capture some local ordering information.

For example, the bigram `free prize` might help distinguish a spam message promising a reward from a simple statement containing the word `free` alone. However, beyond these small sequences, the global order of words and sentence structure remains largely lost. In other words, `CountVectorizer` does not preserve complete word order; it only captures localized patterns defined by the chosen `ngram_range`.

## Using CountVectorizer for the Bag-of-Words Approach

`CountVectorizer` from the `scikit-learn` library efficiently implements the bag-of-words approach. It converts a collection of documents into a matrix of term counts, where each row represents a message and each column corresponds to a term (unigram or bigram). Before transformation, `CountVectorizer` applies tokenization, builds a vocabulary, and then maps each document to a numeric vector.

Key parameters for refining the feature set:

- `min_df=1`: A term must appear in at least one document to be included. While this threshold is set to `1` here, higher values can be used in practice to exclude rare terms.
- `max_df=0.9`: Terms that appear in more than 90% of the documents are excluded, removing overly common words that provide limited differentiation.
- `ngram_range=(1, 2)`: The feature matrix captures individual words and common word pairs by including unigrams and bigrams, potentially improving the model’s ability to detect spam patterns.

Code: python

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the message column
X = vectorizer.fit_transform(df["message"])

# Labels (target variable)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)  # Converting labels to 1 and 0
```

After this step, `X` becomes a numerical feature matrix ready to be fed into a classifier, such as Naive Bayes.

### How CountVectorizer Works

`CountVectorizer` operates in three main stages:

1. `Tokenization`: Splits the text into tokens based on the specified `ngram_range`. For `ngram_range=(1, 2)`, it extracts both unigrams (like "`message`") and bigrams (like "`free prize`").
2. `Building the Vocabulary`: Uses `min_df` and `max_df` to decide which terms to include. Terms that are too rare or common are filtered out, leaving a vocabulary that balances informative and distinctive terms.
3. `Vectorization`: Transforms each document into a vector of term counts. Each vector entry corresponds to a term from the vocabulary, and its value represents how many times that term appears in the document.

### Example with Unigrams

Consider five documents:

1. `The free prize is waiting for you`
2. `The spam message offers a free prize now`
3. `The spam filter might detect this`
4. `The important news says you won a free trip`
5. `The message truly is important`

If we use `ngram_range=(1, 1)` (unigrams only) and `min_df=1`, `max_df=0.9`, the word `The` will be removed from unigram vocabulary by `max_df=0.9` since it appears more than 90% in the documents, leaving the below unigram matrix:

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|0|1|1|1|1|1|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|0|1|0|0|0|0|1|

### Example with Bigrams

Using `ngram_range=(1, 2)`, the final vocabulary includes all of the above unigrams and any valid bigrams containing those unigrams. For instance, `free prize` occurs in Documents 1 and 2. The resulting matrix provides additional context, helping the model differentiate messages more effectively:

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|free prize|prize is|is waiting|waiting for|for you|spam message|message offers|offers a|a free|prize now|spam filter|filter might|might detect|detect this|important news|news says|says you|you won|won a|free trip|message truly|truly is|is important|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|0|0|0|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|1|1|1|1|1|1|0|0|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|1|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|

This feature extraction process, using `CountVectorizer`, has transformed our text data into a resulting matrix provides a concise, numerical representation of each message, ready for training a classification model.

