

After loading the SMS Spam Collection dataset, the next step is preprocessing the text data. Preprocessing standardizes the text, reduces noise, and extracts meaningful features, all of which improve the performance of the Bayes spam classifier. The steps outlined here rely on the `nltk` library for tasks such as tokenization, stop word removal, and stemming.

Before processing any text, you must download the required NLTK data files. These include `punkt` for tokenization and `stopwords` for removing common words that do not contribute to meaning. Ensuring all required resources are available at this stage prevents interruptions during later processing steps.

Code: python

```python
import nltk

# Download the necessary NLTK data files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("=== BEFORE ANY PREPROCESSING ===") 
print(df.head(5))
```

### Lowercasing the Text

`Lowercasing the text` ensures that the classifier treats words equally, regardless of their original casing. By converting all characters to lowercase, the model considers "`Free`" and "`free`" as the `same token`, effectively reducing dimensionality and improving consistency.

Code: python

```python
# Convert all message text to lowercase
df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASING ===")
print(df["message"].head(5))
```

After this step, the `dataset contains uniformly cased text`, preventing the model from assigning different weights to tokens that differ only by letter case.

## Removing Punctuation and Numbers

`Removing unnecessary punctuation and numbers` simplifies the dataset by focusing on meaningful words. However, certain symbols such as `$` and `!` may contain important context in spam messages. For example, `$` might indicate a monetary amount, and `!` might add emphasis.

The code below removes all characters other than lowercase letters, whitespace, dollar signs, or exclamation marks. This balance between cleaning the data and preserving important symbols helps the model concentrate on features relevant to distinguishing spam from ham messages.

Code: python

```python
import re

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))
```

After this step, the text is cleaner, more uniform, and better suited for subsequent preprocessing tasks such as tokenization, stop word removal, or stemming.

## Tokenizing the Text

`Tokenization` divides the message text into individual words or tokens, a crucial step before further analysis. By converting unstructured text into a sequence of words, we prepare the data for operations like removing stop words and applying stemming. `Each token corresponds to a meaningful unit`, allowing downstream processes to operate on smaller, standardized elements rather than entire sentences.

Code: python

```python
from nltk.tokenize import word_tokenize

# Split each message into individual tokens
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))
```

Once tokenized, the dataset contains messages represented as lists of words, ready for additional preprocessing steps that further refine the text data.

## Removing Stop Words

`Stop words` are common words like `and`, `the`, or `is` that often do not add meaningful context. Removing them reduces noise and focuses the model on the words most likely to help distinguish spam from ham messages. By reducing the number of non-informative tokens, we help the model learn more efficiently.

Code: python

```python
from nltk.corpus import stopwords

# Define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["message"].head(5))
```

The token lists are shorter at this stage and contain fewer non-informative words, setting a cleaner stage for future text transformations.

## Stemming

`Stemming` normalizes words by reducing them to their base form (e.g., `running` becomes `run`). This consolidates different forms of the same root word, effectively cutting the vocabulary size and smoothing out the text representation. As a result, the model can better understand the underlying concepts without being distracted by trivial variations in word forms.

Code: python

```python
from nltk.stem import PorterStemmer

# Stem each token to reduce words to their base form
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["message"].head(5))
```

After stemming, the token lists focus on root word forms, further simplifying the text and improving the model’s generalization ability.

## Joining Tokens Back into a Single String

While tokens are useful for manipulation, many machine-learning algorithms and vectorization techniques (e.g., TF-IDF) work best with raw text strings. Rejoining the tokens into a space-separated string restores a format compatible with these methods, allowing the dataset to move seamlessly into the feature extraction phase.

Code: python

```python
# Rejoin tokens into a single string for feature extraction
df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\n=== AFTER JOINING TOKENS BACK INTO STRINGS ===")
print(df["message"].head(5))
```

At this point, the messages are fully preprocessed. Each message is a cleaned, normalized string ready for vectorization and subsequent model training, ultimately improving the classifier’s performance.

