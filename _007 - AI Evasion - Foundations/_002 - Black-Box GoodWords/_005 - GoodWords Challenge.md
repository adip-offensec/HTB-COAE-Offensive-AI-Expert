
You will take a strong spam message and append a compact sequence of legitimate looking words so that a `Multinomial Naive Bayes` model predicts `ham`. The service is black box: you see only labels and spam probabilities from the `/predict` endpoint. When your augmented message satisfies the append only constraint and the word budget, `/submit` returns a static flag.

## Quick start

Start your instance from the platform, and check readiness.

Code: bash

```bash
curl -s "$BASE_URL/health"
```

## API surface

The API is small by design. Each response is JSON.

### GET `/challenge`

Returns the base spam message and the constraints. The server validates that the chosen base message is a high probability spam under the frozen model.

Code: bash

```bash
curl -s "$BASE_URL/challenge" | jq
```

Example response shape:

Code: json

```json
{
  "base_message": "...",
  "max_added_words": 25,
  "target_label": "ham"
}
```

### POST `/predict`

Returns the model’s label and spam probability for arbitrary text. Use this endpoint to estimate single word impact.

Code: bash

```bash
curl -s -X POST "$BASE_URL/predict" \
  -H 'content-type: application/json' \
  -d '{"text": "hello there"}'
```

Example response shape:

Code: json

```json
{
  "label": "spam",
  "spam_probability": 0.9123
}
```

### POST `/submit`

Validates that your `augmented_text` equals the base message followed by at most `max_added_words` additional tokens under the same letter only tokenization used by the model. Returns the flag when the final label is ham.

Code: bash

```bash
curl -s -X POST "$BASE_URL/submit" \
  -H 'content-type: application/json' \
  -d '{"augmented_text": "<base> please thanks meeting hello"}' | jq
```

Example success response shape:

Code: json

```json
{
  "result": "success",
  "details": {"label": "ham", "spam_probability": 0.12, "words_added": 7},
  "flag": "HTB{...}"
}
```

## Minimal Python scaffolds

This short snippet retrieves the challenge and prints the baseline label and probability.

Code: python

```python
import os, requests
host = os.getenv("BASE_URL", "http://127.0.0.1:8080")
ch = requests.get(f"{host}/challenge", timeout=10).json()
base = ch["base_message"]
res = requests.post(f"{host}/predict", json={"text": base}, timeout=15).json()
print(res)
```

This snippet measures single word impacts for a compact vocabulary, then composes an augmentation under the budget.

Code: python

```python
import os, requests, random, numpy as np
random.seed(1337); np.random.seed(1337)
host = os.getenv("BASE_URL", "http://127.0.0.1:8080")
ch = requests.get(f"{host}/challenge", timeout=10).json()
base, budget = ch["base_message"], int(ch["max_added_words"])

def predict(t):
    return requests.post(f"{host}/predict", json={"text": t}, timeout=15).json()

base_p = predict(base)["spam_probability"]
vocab = ["please","thanks","meeting","tomorrow","coffee","home","support","good","great","safe"]
imp = []
for w in vocab:
    p2 = predict(base + " " + w)["spam_probability"]
    imp.append((w, base_p - p2))
imp.sort(key=lambda x: x[1], reverse=True)
top = [w for w, d in imp if d > 0][: max(2*budget, 20)]

aug = base
for i, w in enumerate(top, 1):
    if i > budget: break
    aug = aug + " " + w
    lab = predict(aug)["label"]
    if lab == "ham":
        break

print(requests.post(f"{host}/submit", json={"augmented_text": aug}, timeout=15).json())
```

1. What is the flag you get after a successful attack?
	1. Execute the below script

```python
import requests

host = "http://154.57.164.67:31276"

# Get the challenge
ch = requests.get(f"{host}/challenge").json()
base = ch["base_message"]
budget = ch["max_added_words"]

def predict(t):
    return requests.post(f"{host}/predict", json={"text": t}).json()

base_res = predict(base)
print("Baseline:", base_res)
base_p = base_res["spam_probability"]

# Large list of potential ham words, including common English words and SMS-like terms
vocab = list(set([
    "please", "thanks", "meeting", "tomorrow", "coffee", "home", "support", "good", "great", "safe",
    "hello", "hi", "dear", "regards", "best", "sincerely", "thank", "you", "for", "your",
    "time", "help", "assistance", "team", "project", "work", "office", "call", "phone", "email",
    "message", "note", "letter", "family", "friend", "love", "care", "health", "well", "being",
    "morning", "afternoon", "evening", "day", "week", "month", "year", "plan", "schedule", "appointment",
    "school", "teacher", "student", "class", "lesson", "book", "read", "write",
    "paper", "computer", "web", "site", "page",
    "the", "and", "to", "of", "a", "in", "is", "it", "that",
    "he", "was", "on", "are", "with", "as", "I", "his", "they",
    "be", "at", "one", "have", "this", "from", "or", "had", "by", "but",
    "some", "what", "there", "we", "can", "out", "other", "were", "all",
    "when", "up", "use", "how", "said", "an", "each", "she",
    "which", "do", "their", "if", "will", "way", "about", "many",
    "then", "them", "would", "like", "so", "these", "her", "long",
    "make", "thing", "see", "him", "two", "has", "look", "more",
    "could", "go", "come", "did", "my", "no", "most", "number", "who",
    "over", "know", "than", "first", "people", "may", "down", "side",
    "been", "now", "find", "any", "new", "part", "take", "get",
    "place", "made", "where", "after", "back", "little", "only",
    "man", "year", "came", "show", "every", "me", "give", "our",
    "under", "name", "very", "through", "just", "form", "much",
    "think", "say", "low", "line", "before", "turn", "cause",
    "same", "mean", "move", "right", "boy", "old", "too",
    "does", "tell", "set", "three", "want", "well",
    "also", "play", "small", "end", "put", "read", "hand",
    "large", "add", "even", "land", "here", "must", "big",
    "high", "such", "follow", "act", "why", "ask", "men", "change",
    "went", "light", "kind", "off", "need", "house", "try",
    "us", "again", "point", "mother", "world", "near", "build",
    "self", "father", "head", "stand", "own", "should",
    "country", "found", "answer", "grow", "study", "still",
    "learn", "food", "four", "thought", "let",
    "keep", "eye", "never", "last", "door", "between", "city",
    "since", "hard", "start", "might", "story", "far",
    "draw", "left", "late", "run", "don't", "while",
    "close", "night", "real", "life", "few", "stop", "open", "seem",
    "together", "next", "white", "children", "begin", "got", "walk",
    "example", "paper", "always", "music", "those",
    "both", "book", "letter", "until", "car",
    "care", "second", "group", "took", "eat",
    "room", "friend", "idea", "once",
    "sure", "watch", "color", "face",
    "main", "enough", "girl", "young", "ready",
    "red", "list", "though", "feel", "talk", "bird",
    "soon", "body", "dog", "direct", "leave",
    "state", "black", "short", "class",
    "question", "happen", "complete", "area", "half",
    "order", "told", "knew", "pass",
    "top", "whole", "heard", "best", "hour",
    "better", "during", "am", "remember", "step",
    "early", "hold", "interest", "reach", "fast",
    "five", "listen", "table", "travel", "less", "morning",
    "ok", "yes", "later", "coming", "going", "see", "meet",
    "u", "ur", "wat", "hey", "yo", "doing", "where", "how", "when", "what", "why",
    "im", "you're", "don't", "can't", "won't"
]))

# Measure impact of each word
imp = []
for w in vocab:
    try:
        p2 = predict(base + " " + w)["spam_probability"]
        delta = base_p - p2
        if delta > 0:
            imp.append((w, delta))
    except Exception as e:
        print(f"Error for word {w}: {e}")

# Sort by highest impact
imp.sort(key=lambda x: x[1], reverse=True)

# Take top candidates, more than budget
top_words = [w for w, d in imp][:budget * 2]

# Greedily add top words
aug = base
added_words = []
for w in top_words:
    if len(added_words) >= budget:
        break
    test_aug = aug + " " + w
    res = predict(test_aug)
    if res["label"] == "ham":
        aug = test_aug
        added_words.append(w)
        break
    else:
        aug = test_aug
        added_words.append(w)

print("Augmented text:", aug)
print("Added words:", len(added_words))

# If not ham yet, try adding the best word repeatedly
if predict(aug)["label"] != "ham" and len(added_words) < budget:
    if imp:
        best_w, _ = imp[0]
        while len(added_words) < budget:
            aug += " " + best_w
            added_words.append(best_w)
            res = predict(aug)
            if res["label"] == "ham":
                break

# Submit
submit_res = requests.post(f"{host}/submit", json={"augmented_text": aug}).json()
print(submit_res)

if "flag" in submit_res:
    print("Flag:", submit_res["flag"])
```

The output is below

```
Baseline: {'label': 'spam', 'spam_probability': 0.9999999983049008}
Augmented text: England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+ later ask say doing home said sure
Added words: 7
{'result': 'success', 'details': {'label': 'ham', 'spam_probability': 0.11316731988689203, 'words_added': 7}, 'flag': 'HTB{g00dw0rd5_15_1n73r3571ng}'}
Flag: HTB{g00dw0rd5_15_1n73r3571ng}
```

