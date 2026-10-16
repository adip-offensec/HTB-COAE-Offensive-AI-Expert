
# Skills Assessment: Feature Obfuscation Attack

Demonstrate your skills in executing `GoodWords` attacks against two `Naive Bayes` classifiers in this two-phase assessment. First, you'll convert positive movie reviews into negative sentiment using white-box model access. Next, you'll turn negative reviews into positive ones using only black-box API queries. Successfully complete both phases to capture the flag.

### Phase 1: White-Box Attack

- `Objective`: Flip the sentiment of 10 positive movie reviews to negative.
- `Resources`: Full access to download and analyze the model.
- `Constraints`: Add up to a maximum of 30 words per review.

### Phase 2: Black-Box Attack

- `Objective`: Flip the sentiment of 10 negative movie reviews to positive.
- `Resources`: Limited to black-box API queries; no direct model access.
- `Constraints`: Add up to a maximum of 40 words per review.

## Quick Start

First verify your instance is running and set the base URL for convenience:

Code: bash

```bash
export BASE_URL="http://your-instance:8080"
curl -s "$BASE_URL/health" | jq
```

The health endpoint confirms readiness. Wait a few moments if you see connection errors. The expected response is `{"status": "healthy", "service": "skills_assessment_lab"}`.

# API Documentation

## Phase 1: White-Box Endpoints

### GET `/challenge/whitebox`

Retrieves the white-box challenge data:

Code: bash

```bash
curl -s "$BASE_URL/challenge/whitebox" | jq
```

Response structure:

Code: json

```json
{
  "phase": "whitebox",
  "description": "Flip 10 positive reviews to negative sentiment",
  "reviews": [
    {"id": "wb_0", "text": "...", "target_sentiment": "negative"},
    ...
  ],
  "max_added_words": 30,
  "model_endpoint": "/model/download",
  "submit_endpoint": "/submit/whitebox"
}
```

### GET `/model/download`

Downloads the complete model bundle for analysis:

Code: bash

```bash
curl -s "$BASE_URL/model/download" -o model.pkl
```

The pickle file contains the trained classifier, vectorizer, feature names, and class labels.

### POST `/submit/whitebox`

Submits your solutions for validation:

Code: bash

```bash
curl -s -X POST "$BASE_URL/submit/whitebox" \
  -H 'Content-Type: application/json' \
  -d '{
    "solutions": [
      {"id": "wb_0", "augmented_text": "original text plus added words"},
      {"id": "wb_1", "augmented_text": "another review with additions"}
    ]
  }' | jq
```

## Phase 2: Black-Box Endpoints

### GET `/challenge/blackbox`

Retrieves the black-box challenge data:

Code: bash

```bash
curl -s "$BASE_URL/challenge/blackbox" | jq
```

Response format matches white-box but with different reviews and increased word budget (40 words).

### POST `/predict`

Queries the model for predictions without accessing internals:

Code: bash

```bash
curl -s -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text": "sample text to classify"}' | jq
```

Returns:

Code: json

```json
{
  "label": "positive",
  "negative_probability": 0.234,
  "positive_probability": 0.766
}
```

### POST `/submit/blackbox`

Submits black-box solutions using the same format as white-box.

## GET `/status`

Check your current progress:

Code: bash

```bash
curl -s "$BASE_URL/status" | jq
```

## Python Starter Code

These minimal scaffolds help you begin exploring the challenge:

### Initial Setup

Code: python

```python
import os, requests, pickle
import numpy as np

np.random.seed(1337)  # For reproducibility
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080")

def get_challenge(phase):
    """Fetch challenge data for specified phase"""
    r = requests.get(f"{BASE_URL}/challenge/{phase}")
    return r.json()

def submit_solutions(phase, solutions):
    """Submit solutions for validation"""
    r = requests.post(
        f"{BASE_URL}/submit/{phase}",
        json={"solutions": solutions}
    )
    return r.json()
```

### White-Box Model Loading

Code: python

```python
def load_model():
    """Download and load the model bundle"""
    # Download model
    r = requests.get(f"{BASE_URL}/model/download")
    with open("model.pkl", "wb") as f:
        f.write(r.content)

    # Load bundle
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)

    return bundle

# Examine model structure
bundle = load_model()
print(f"Keys in bundle: {bundle.keys()}")
print(f"Model type: {type(bundle['classifier'])}")
print(f"Number of features: {len(bundle['feature_names'])}")
```

### Black-Box Prediction Query

Code: python

```python
def predict(text):
    """Query prediction API"""
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    return r.json()

# Test baseline probability
sample = "This movie was terrible and boring"
result = predict(sample)
print(f"Label: {result['label']}")
print(f"Positive probability: {result['positive_probability']:.3f}")
```

## Success Criteria

Your goal is to successfully flip all 10 reviews in each phase while respecting the constraints:

- White-box: All 10 positive reviews must predict as negative
- Black-box: All 10 negative reviews must predict as positive
- Both phases must complete to receive the flag

The server will validate each submission and provide feedback on which reviews succeeded or failed. Use this information to refine your approach.

1. What is the flag value you get from the instance api after completing the skills assessment?
	1. Execute the below script

```python
import os
import requests
import pickle
import numpy as np
from sklearn.base import is_classifier  # Just to check

np.random.seed(1337)  # For reproducibility
BASE_URL = "http://154.57.164.65:31204"  # Replace with your instance if different

def get_challenge(phase):
    """Fetch challenge data for specified phase"""
    r = requests.get(f"{BASE_URL}/challenge/{phase}")
    return r.json()

def submit_solutions(phase, solutions):
    """Submit solutions for validation"""
    r = requests.post(
        f"{BASE_URL}/submit/{phase}",
        json={"solutions": solutions}
    )
    return r.json()

def load_model():
    """Download and load the model bundle"""
    r = requests.get(f"{BASE_URL}/model/download")
    with open("model.pkl", "wb") as f:
        f.write(r.content)
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

def predict_blackbox(text):
    """Query prediction API for black-box"""
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    return r.json()

# Phase 1: White-Box Attack
print("Starting Phase 1: White-Box")
challenge_wb = get_challenge("whitebox")
reviews_wb = challenge_wb["reviews"]
max_added_wb = challenge_wb["max_added_words"]

bundle = load_model()
classifier = bundle["classifier"]
vectorizer = bundle["vectorizer"]
feature_names = bundle["feature_names"]

# Determine class indices
classes = classifier.classes_
neg_idx = np.where(classes == "negative")[0][0]
pos_idx = np.where(classes == "positive")[0][0]

# Find best negative word: max log P(w|neg) - log P(w|pos)
neg_log_prob = classifier.feature_log_prob_[neg_idx]
pos_log_prob = classifier.feature_log_prob_[pos_idx]
diff = neg_log_prob - pos_log_prob
best_idx = np.argmax(diff)
best_neg_word = feature_names[best_idx]
print(f"Best negative word: {best_neg_word}")

solutions_wb = []
for review in reviews_wb:
    id_wb = review["id"]
    text = review["text"]
    augmented = text
    added = 0
    vect = vectorizer.transform([augmented])
    pred = classifier.predict(vect)[0]
    while pred != "negative" and added < max_added_wb:
        augmented += " " + best_neg_word
        added += 1
        vect = vectorizer.transform([augmented])
        pred = classifier.predict(vect)[0]
    solutions_wb.append({"id": id_wb, "augmented_text": augmented})
    print(f"Review {id_wb} flipped with {added} words added")

submit_resp_wb = submit_solutions("whitebox", solutions_wb)
print("White-Box Submit Response:", submit_resp_wb)

# Phase 2: Black-Box Attack
print("Starting Phase 2: Black-Box")
challenge_bb = get_challenge("blackbox")
reviews_bb = challenge_bb["reviews"]
max_added_bb = challenge_bb["max_added_words"]

# Candidate positive words
positive_candidates = [
    "great", "excellent", "wonderful", "amazing", "fantastic",
    "brilliant", "superb", "outstanding", "terrific", "awesome"
]

# Find best positive word by querying single word positive prob
best_pos_prob = 0
best_pos_word = None
for word in positive_candidates:
    result = predict_blackbox(word)
    pos_prob = result["positive_probability"]
    if pos_prob > best_pos_prob:
        best_pos_prob = pos_prob
        best_pos_word = word
print(f"Best positive word: {best_pos_word} with prob {best_pos_prob}")

solutions_bb = []
for review in reviews_bb:
    id_bb = review["id"]
    text = review["text"]
    augmented = text
    added = 0
    result = predict_blackbox(augmented)
    label = result["label"]
    while label != "positive" and added < max_added_bb:
        augmented += " " + best_pos_word
        added += 1
        result = predict_blackbox(augmented)
        label = result["label"]
    solutions_bb.append({"id": id_bb, "augmented_text": augmented})
    print(f"Review {id_bb} flipped with {added} words added")

submit_resp_bb = submit_solutions("blackbox", solutions_bb)
print("Black-Box Submit Response:", submit_resp_bb)

# Check status for flag
status_r = requests.get(f"{BASE_URL}/status")
print("Status Response:", status_r.json())
```

The output for the script is below including the flag

```
Starting Phase 1: White-Box
/home//miniconda3/envs/ai/lib/python3.11/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator MultinomialNB from version 1.7.2 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
/home//miniconda3/envs/ai/lib/python3.11/site-packages/sklearn/base.py:463: InconsistentVersionWarning: Trying to unpickle estimator CountVectorizer from version 1.7.2 when using version 1.8.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
Best negative word: hours life
Review wb_0 flipped with 6 words added
Review wb_1 flipped with 3 words added
Review wb_2 flipped with 3 words added
Review wb_3 flipped with 4 words added
Review wb_4 flipped with 6 words added
Review wb_5 flipped with 10 words added
Review wb_6 flipped with 5 words added
Review wb_7 flipped with 4 words added
Review wb_8 flipped with 21 words added
Review wb_9 flipped with 7 words added
White-Box Submit Response: {'completed': 9, 'required': 10, 'results': [{'id': 'wb_0', 'success': True, 'words_added': 12}, {'id': 'wb_1', 'success': True, 'words_added': 6}, {'id': 'wb_2', 'success': True, 'words_added': 6}, {'id': 'wb_3', 'success': True, 'words_added': 8}, {'id': 'wb_4', 'success': True, 'words_added': 12}, {'id': 'wb_5', 'success': True, 'words_added': 20}, {'id': 'wb_6', 'success': True, 'words_added': 10}, {'id': 'wb_7', 'success': True, 'words_added': 8}, {'error': 'Too many words added: 42 > 30', 'id': 'wb_8'}, {'id': 'wb_9', 'success': True, 'words_added': 14}]}
Starting Phase 2: Black-Box
Best positive word: superb with prob 0.8451474677485522
Review bb_0 flipped with 18 words added
Review bb_1 flipped with 28 words added
Review bb_2 flipped with 17 words added
Review bb_3 flipped with 15 words added
Review bb_4 flipped with 4 words added
Review bb_5 flipped with 27 words added
Review bb_6 flipped with 33 words added
Review bb_7 flipped with 4 words added
Review bb_8 flipped with 12 words added
Review bb_9 flipped with 5 words added
Black-Box Submit Response: {'flag': 'HTB{f34tur3_0bfu5c4t10n_m45t3r3d}', 'message': 'Congratulations! Both phases complete!', 'phase_complete': True, 'results': [{'id': 'bb_0', 'success': True, 'words_added': 18}, {'id': 'bb_1', 'success': True, 'words_added': 28}, {'id': 'bb_2', 'success': True, 'words_added': 17}, {'id': 'bb_3', 'success': True, 'words_added': 15}, {'id': 'bb_4', 'success': True, 'words_added': 4}, {'id': 'bb_5', 'success': True, 'words_added': 27}, {'id': 'bb_6', 'success': True, 'words_added': 33}, {'id': 'bb_7', 'success': True, 'words_added': 4}, {'id': 'bb_8', 'success': True, 'words_added': 12}, {'id': 'bb_9', 'success': True, 'words_added': 5}]}
Status Response: {'blackbox_completed': 10, 'blackbox_required': 10, 'total_queries': 183, 'whitebox_completed': 9, 'whitebox_required': 10}
```

