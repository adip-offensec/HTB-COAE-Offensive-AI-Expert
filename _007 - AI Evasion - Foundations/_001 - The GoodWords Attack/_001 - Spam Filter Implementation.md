
To demonstrate the GoodWords attack, we need a target to attack. To this end, we will build a `Naive Bayes` spam classifier trained on real SMS data.

## Library Installation

This module uses the HTB Evasion Library which provides common utilities. Install it first:

Code: bash

```bash
# Install the AI Library (or update it)
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

## Initial Setup and Dependencies

First, we'll import the necessary libraries and set seeds for reproducibility. We'll organize our imports into three groups, standard library, third‑party packages, and our custom HTB utilities:

Code: python

```python
import json
import pickle
import random
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

# Reproducibility
random.seed(1337)
np.random.seed(1337)
```

Next, we'll import the data science and machine learning libraries used to build the classifier:

Code: python

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
```

We'll also import the HTB library utilities for consistent styling and helper functions:

Code: python

```python
from htb_ai_library import (
    AZURE,
    HACKER_GREY,
    HTB_GREEN,
    MALWARE_RED,
    NODE_BLACK,
    NUGGET_YELLOW,
    WHITE,
    AQUAMARINE,
    load_model,
    save_model,
)
```

The seed value `1337` ensures our results remain reproducible across runs, which is essential for verifying attack effectiveness consistently. The HTB library provides both functionality for model persistence and consistent color constants that follow the HTB visual theme. Using `pathlib.Path` instead of string concatenation handles Windows and Linux path differences automatically. For example, `Path("data") / "file.txt"` produces `data\file.txt` on Windows and `data/file.txt` on Linux.

## Loading the SMS Spam Dataset

We'll use the SMS Spam Collection dataset from the UCI Machine Learning Repository, which contains 5,574 labeled messages. To avoid unnecessary downloads, implement a caching mechanism that checks for local copies first:

Code: python

```python
print("\n[*] Loading SMS Spam Dataset...")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
dataset_path = data_dir / "sms_spam.csv"
```

The `Path("data").mkdir(exist_ok=True)` creates our data directory if it doesn't exist, preventing errors on first run. The `exist_ok=True` parameter means the operation succeeds even if the directory already exists, making our code idempotent.

### Checking for Cached Data

The UCI dataset weighs 198KB compressed but requires network round-trips and extraction overhead. Caching eliminates this cost on subsequent runs:

Code: python

```python
if dataset_path.exists():
    print(f"[+] Using cached dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
else:
    print("[*] Downloading from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = data_dir / "sms_spam.zip"

    urllib.request.urlretrieve(url, zip_path)
```

The file system check happens in microseconds versus seconds for network operations. On a typical connection, the initial download takes 2-3 seconds, while cached reads complete in under 50ms. This 40× speedup becomes significant during iterative development where you might run the notebook dozens of times while tuning hyperparameters or testing attack variations. The cached CSV format also loads faster than parsing the original tab-separated text, adding another 2-3× performance gain from pandas' optimized CSV reader.

### Extracting and Processing the Download

When we don't have a cached copy, we'll download and process the original format. The UCI dataset comes as a zip file containing tab-separated values that we need to extract and parse:

Code: python

```python
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("utf-8").strip() for line in f]

    # Parse tab-separated format
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            data.append({"label": parts[0].lower(), "message": parts[1]})

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    zip_path.unlink()
    print(f"[+] Dataset saved to {dataset_path}")
```

The in-memory extraction avoids filesystem I/O overhead. Instead of unzipping all files to disk (creating temporary files, writing data, then reading it back), we stream directly from the compressed archive into memory. This reduces the operation from disk write + disk read to a single decompression pass, cutting processing time by roughly half. The validation `len(parts) == 2` guards against corrupted records. Real-world datasets sometimes contain blank lines, partial records from transmission errors, or malformed entries. Filtering these prevents downstream errors when the classifier expects consistent label-message pairs.

### Understanding the Dataset

Now let's explore the dataset structure and class distribution to understand what we're working with:

Code: python

```python
print(f"[+] Loaded {len(df)} messages")
print(f"    Spam: {sum(df['label'] == 'spam')}")
print(f"    Ham: {sum(df['label'] == 'ham')}")
```

Code: txt

```txt
[*] Loading SMS Spam Dataset...
[+] Using cached dataset: data/sms_spam.csv
[+] Loaded 5572 messages
    Spam: 747
    Ham: 4825
```

We observe significant class imbalance with 747 spam messages (13.4%) and 4,825 ham messages (86.6%). This mirrors real-world email distributions where legitimate messages vastly outnumber spam. The imbalance makes our future attacks more realistic, as classifiers trained on skewed data often exhibit exploitable biases toward the majority class.

Let's examine a few message samples to see the patterns the classifier needs to learn:

Code: python

```python
print("\n[*] Sample messages:")
print("\nSPAM samples:")
for msg in df[df['label'] == 'spam']['message'].head(3):
    print(f"  - {msg[:80]}...")
print("\nHAM samples:")
for msg in df[df['label'] == 'ham']['message'].head(3):
    print(f"  - {msg[:80]}...")
```

Code: txt

```txt
SPAM samples:
  - Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 8...
  - FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some...
  - WINNER!! As a valued network customer you have been selected to receivea £900 pr...

HAM samples:
  - Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
  - Ok lar... Joking wif u oni......
  - U dun say so early hor... U c already then say......
```

The spam samples exhibit classic indicators: "Free", "WINNER!!", and currency symbols like "£900". In contrast, ham samples contain informal conversational language with abbreviations and casual spelling. These distinct patterns will become the features our classifier uses for discrimination.

## Text Preprocessing

Now we need to transform raw messages into a consistent format suitable for machine learning. We'll use two levels of cleaning, minimal cleaning that preserves spam indicators, and final cleaning for vectorization.

### Minimal Cleaning for Analysis

Let's begin with minimal cleaning that preserves the characteristics spam filters look for:

Code: python

```python
import html as html_module
import unicodedata
```

We implement a minimal cleaning function that preserves spam indicators while normalizing encoding and whitespace:

Code: python

```python
def minimal_clean(text):
    """
    Minimal cleaning that preserves spam indicators.

    Parameters: text (str) raw SMS message
    Returns: str cleaned text with entities decoded, unicode normalized, and
             whitespace collapsed while keeping informative symbols.
    """
    # Decode HTML entities (e.g., &amp; -> &)
    text = html_module.unescape(text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)

    return text.strip()
```

The `html_module.unescape()` function converts HTML entities like `&amp;` to their actual characters (`&`). This handles messages that might have been scraped from web sources. Unicode normalization with `'NFKC'` mode converts characters to their canonical composed form. For example, the ligature "ﬁ" becomes "fi", preventing identical content from being treated differently due to encoding variations.

We handle whitespace normalization using three separate regex patterns. The pattern `r'\s+'` matches any sequence of whitespace (spaces, tabs) and replaces it with a single space. We handle newlines and carriage returns separately to ensure consistent processing across different text sources.

### Final Cleaning for Vectorization

Next, we prepare text for the count vectorizer:

Code: python

```python
def clean_text(text):
    """
    Final cleaning for vectorization.

    Converts to lowercase and removes only problematic characters so that
    informative symbols remain available to the vectorizer.

    Parameters:
        text (str): Preprocessed message from `minimal_clean`.

    Returns:
        str: Normalized, whitespace‑collapsed text ready for tokenization.
    """
    text = text.lower()
    # Keep numbers, currency symbols, punctuation - they're spam features!
    # Only remove truly problematic characters
    text = re.sub(r'[^\w\s£$€¥!?.,;:\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

Converting to lowercase ensures "FREE" and "free" become the same feature, preventing duplication in our vocabulary. The character filtering regex `[^\w\s£$€¥!?.,;:\'\"-]` preserves important features by specifying what to keep. We retain `\w` (word characters), `\s` (whitespace), and specific punctuation because these carry discriminative power. Excessive exclamation marks ("!!!") signal urgency often found in spam, while currency symbols ("£900 prize") indicate monetary offers, another spam characteristic.

### Applying Preprocessing

Let's apply both cleaning stages to the dataset:

Code: python

```python
print("[*] Applying minimal text cleaning (preserving spam indicators)...")
df['preprocessed'] = df['message'].apply(minimal_clean)

# Apply final cleaning for vectorization
df['clean_message'] = df['preprocessed'].apply(clean_text)
```

The `apply()` method vectorizes the cleaning operation across all messages efficiently. We store both versions - `preprocessed` for analysis and `clean_message` for model training.

Let's inspect what preprocessing preserved:

Code: python

```python
print("\n[*] Sample spam messages with preserved features:")
spam_samples = df[df['label'] == 'spam'].sample(3, random_state=42)
for idx, row in spam_samples.iterrows():
    msg = row['preprocessed'][:100] + "..." if len(row['preprocessed']) > 100 else row['preprocessed']
    print(f"  - {msg}")
```

Code: txt

```txt
[*] Sample spam messages with preserved features:
  - Summers finally here! Fancy a chat or flirt with sexy singles in yr area? To get MATCHED up just rep...
  - This is the 2nd time we have tried 2 contact u. U have won the 750 Pound prize. 2 claim is easy, cal...
  - Get ur 1st RINGTONE FREE NOW! Reply to this msg with TONE. Gr8 TOP 20 tones to your phone every week...
```

Notice how preprocessing preserves spam indicators like excessive punctuation, currency references ("750 Pound prize"), and uppercase words ("FREE NOW!", "MATCHED"). These features are important for accurate classification.

### Removing Duplicates and Empty Messages

Let's clean the dataset by removing duplicate and empty entries that could skew training:

Code: python

```python
# Remove only exact duplicates
original_size = len(df)
df = df.drop_duplicates(subset=['label', 'clean_message'])
print(f"\n[+] Removed {original_size - len(df)} duplicates")

# Remove empty messages
before_empty = len(df)
df = df[df['clean_message'].str.len() > 0]
print(f"[+] Removed {before_empty - len(df)} empty messages")
```

Code: txt

```txt
[+] Removed 419 duplicates
[+] Removed 0 empty messages

[+] Final dataset: 5153 messages
    Spam: 638 (12.4%)
    Ham: 4515 (87.6%)
```

The `drop_duplicates()` call with `subset=['label', 'clean_message']` removes messages with identical text and label after cleaning, eliminating 419 duplicates. We only remove exact duplicates where both the label and cleaned text match, preserving legitimate cases where the same text might appear with different labels.

The empty message check `df['clean_message'].str.len() > 0` ensures we don't train on contentless samples, though our dataset has none. After cleaning, we have 5,153 unique messages with a 12.4% spam ratio, down from the original 13.4%. The higher duplicate rate in spam messages reveals their repetitive nature, a characteristic we'll exploit in our attack.

## Creating Train and Test Sets

Let's split the data into training and testing sets. Use stratified sampling to maintain the class distribution across both sets:

Code: python

```python
X = df['clean_message'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[+] Data split:")
print(f"    Training: {len(X_train)} messages")
print(f"    Testing: {len(X_test)} messages")
```

Code: txt

```txt
[+] Data split:
    Training: 4122 messages
    Testing: 1031 messages
```

Using `.values` extracts NumPy arrays from our DataFrame for more efficient processing by scikit-learn. The `stratify=y` parameter maintains the 12.4% spam ratio in both train and test sets. Without stratification, random sampling could create imbalanced splits that would distort our performance metrics.

Our 80-20 split yields 4,122 training messages and 1,031 testing messages. This provides sufficient data for the model to learn spam patterns while keeping enough test samples for reliable performance evaluation.

## Training the Naive Bayes Classifier

We'll now train our `MultinomialNB` classifier with a count vectorizer for feature extraction. To ensure reproducibility, we'll also implement model persistence:

### Setting Up Model Training

Let's check for a saved model to avoid retraining, or train a new one if needed:

Code: python

```python
print("\n[*] Training Naive Bayes classifier...")

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "spam_classifier.pkl"

if model_path.exists():
    print(f"[+] Loading saved model from {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data['vectorizer']
        classifier = saved_data['classifier']

    # Transform data using existing vocabulary
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
```

Using `pickle` for model persistence lets us resume analysis without retraining, saving computation time. When loading a saved model, we apply `transform()` instead of `fit_transform()` to use the exact vocabulary from the original training. This maintains feature space consistency by ignoring any new words not in the original vocabulary.

### Training a New Model

When starting fresh, we'll configure our vectorizer to capture spam-specific patterns:

Code: python

```python
else:
    # Configure vectorizer to capture spam patterns
    vectorizer = CountVectorizer(
        max_features=3000,
        token_pattern=r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+',
        lowercase=True,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
```

Let's train the classifier and save both the vectorizer and model together:

Code: python

```python
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Save model for reproducibility
    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': classifier}, f)
    print(f"[+] Model saved to {model_path}")
```

Our `CountVectorizer` uses a custom `token_pattern` regex to capture spam-specific features. The pattern `r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+'` uses alternation (`|`) to match multiple token types. We capture standard words with `\b\w+\b`, currency symbols with `[£$€¥]+`, and numbers with `\d+`. The patterns `!!+`, `\?\?+`, and `\.\.+` specifically target repeated punctuation common in spam.

We limit vocabulary to 3000 features with `max_features=3000`, preventing overfitting while maintaining expressiveness. Removing English stop words eliminates non-discriminative terms like "the" and "is" that appear equally in both classes.

### Evaluating Model Performance

Let's evaluate the classifier's performance on both training and test sets to check for overfitting:

Code: python

```python
# Calculate accuracy scores
train_acc = classifier.score(X_train_vec, y_train)
test_acc = classifier.score(X_test_vec, y_test)
print(f"[+] Training accuracy: {train_acc:.4f}")
print(f"[+] Testing accuracy: {test_acc:.4f}")

# Get detailed predictions
y_pred = classifier.predict(X_test_vec)
print("\n[*] Classification Report:")
print(classification_report(y_test, y_pred))
```

Code: txt

```txt
[*] Training Naive Bayes classifier...
[+] Loading saved model from models/spam_classifier.pkl
[+] Training accuracy: 0.9922
[+] Testing accuracy: 0.9864

[*] Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       903
        spam       0.95      0.94      0.94       128

    accuracy                           0.99      1031
   macro avg       0.97      0.97      0.97      1031
weighted avg       0.99      0.99      0.99      1031
```

The `.score()` method calculates accuracy as the fraction of correct predictions. We achieve 99.22% training accuracy and 98.64% test accuracy. The minimal gap (0.58%) confirms our model generalizes well without memorizing the training data.

Examining the classification report shows us class-specific performance. Ham detection achieves 99% precision and recall, correctly identifying legitimate messages with minimal errors. Spam detection shows 95% precision (when we flag spam, we're right 95% of the time) and 94% recall (we catch 94% of all spam).

The slightly lower spam metrics result from class imbalance. With fewer spam examples to learn from, the model has less confidence about spam patterns. This realistic performance level makes our classifier an ideal target for demonstrating the GoodWords attack's effectiveness.

