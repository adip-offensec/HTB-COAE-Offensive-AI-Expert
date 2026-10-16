The `IMDB dataset` introduced by Maas et al. (2011) provides a collection of movie reviews extracted from the Internet Movie Database, annotated for `sentiment analysis`. It includes 50,000 reviews split evenly into training and test sets, and its carefully curated mixture of positive and negative examples allows researchers to benchmark and improve various natural language processing techniques. The `IMDB dataset` has influenced subsequent work in developing vector-based word representations and remains a popular baseline resource for evaluating classification performance and model architectures in sentiment classification tasks ([Maas et al., 2011](http://www.aclweb.org/anthology/P11-1015)).

Your goal is to train a model that can predict whether a movie review is positive (`1`) or negative (`0`). You can download the dataset from the question, or [from here](https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip).

Out of interest, these exact same techniques can be applied into things such as text moderation for instance.

---

To evaluate your model, upload it to the evaluation portal running on the Playground VM. If you are not currently using the Playground VM, you can initialize it at the bottom of the page.

If you have the Playground VM running, you can use this Python script to upload your model from Jupyter directly. Once evaluated, if your model meets the required performance criteria, you will receive a flag value. This flag can be used to answer the question or verify the model’s success.

Code: python

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:5000/api/upload"

# Path to the model file you want to upload
model_file_path = "skills_assessment.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

If you are working from your own machine, ensure you have configured the HTB VPN to connect to the remote VM and spawned it. After connecting, access the model upload portal by navigating to `http://VM-IP:5000/` in your browser and then uploading your model.

1. What is the flag you get from submitting a good model for evaluation?

Run the below code in JupyterLab

```python
#!/usr/bin/env python3
"""
spam_style_imdb_train_and_upload.py

Spam-style sentiment classifier:
- CountVectorizer + MultinomialNB
- GridSearchCV over alpha
- Saves Pipeline with steps named: "vectorizer" and "classifier"
- Uploads to HTB evaluator

Dataset:
- Downloads https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip
- Extracts into ./data/skills_assessment_data/
- Loads train.json (preferred) for training
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


# -------------------- Config --------------------
DATA_URL = "https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip"
UPLOAD_URL = "http://10.129.83.233:5000/api/upload"

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "skills_assessment_data.zip"
EXTRACT_DIR = DATA_DIR / "skills_assessment_data"

MODEL_FILENAME = "skills_assessment.joblib"


# -------------------- Download + Extract --------------------
def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None

        downloaded = 0
        start = time.time()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    speed = downloaded / max(time.time() - start, 1e-6) / (1024 * 1024)
                    print(
                        f"\rDownloading: {pct:6.2f}% "
                        f"({downloaded/1024/1024:.2f} MB) "
                        f"{speed:.2f} MB/s",
                        end=""
                    )
        print()


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


# -------------------- Dataset loading (JSON) --------------------
def normalize_label(val: Any) -> Optional[int]:
    """
    Accepts labels like:
      0/1
      "pos"/"neg"
      "positive"/"negative"
      rating 1..10 (>=6 -> pos)
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        if int(val) in (0, 1):
            return int(val)
        if 1 <= float(val) <= 10:
            return 1 if float(val) >= 6 else 0

    s = str(val).strip().lower()
    if s in ("0", "neg", "negative", "bad"):
        return 0
    if s in ("1", "pos", "positive", "good"):
        return 1
    if "neg" in s:
        return 0
    if "pos" in s:
        return 1

    return None


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "records", "examples"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise RuntimeError(f"Unrecognized JSON format in {path}")


def load_training_data(extract_root: Path) -> Tuple[List[str], List[int]]:
    # Prefer train.json; fall back to any json that parses
    preferred = []
    for name in ("train.json", "data.json", "dataset.json", "test.json"):
        p = extract_root / name
        if p.exists():
            preferred.append(p)

    candidates = preferred + sorted([p for p in extract_root.rglob("*.json") if p not in preferred])
    if not candidates:
        raise RuntimeError("No .json files found in extracted dataset folder")

    text_keys = ["review", "text", "content", "sentence"]
    label_keys = ["label", "sentiment", "target", "class", "y", "rating"]

    for f in candidates:
        try:
            records = load_json_records(f)
        except Exception:
            continue

        X: List[str] = []
        y: List[int] = []

        for r in records:
            if not isinstance(r, dict):
                continue

            text = None
            lab_raw = None

            for k in text_keys:
                if k in r:
                    text = r[k]
                    break
            for k in label_keys:
                if k in r:
                    lab_raw = r[k]
                    break

            lab = normalize_label(lab_raw)
            if text is None or lab is None:
                continue

            X.append(str(text))
            y.append(lab)

        if len(X) >= 5000:
            print(f"[+] Loaded training data from: {f} (samples={len(X):,})")
            return X, y

    raise RuntimeError("Could not parse a usable labeled dataset from JSON files")


# -------------------- Train + Save + Upload --------------------
def main() -> None:
    print("[+] Starting spam-style sentiment pipeline")

    # Download
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        print("[+] Downloading dataset...")
        download_file(DATA_URL, ZIP_PATH)
    else:
        print(f"[+] Zip already exists: {ZIP_PATH}")

    # Extract
    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.rglob("*")):
        print("[+] Extracting dataset...")
        extract_zip(ZIP_PATH, EXTRACT_DIR)
    else:
        print(f"[+] Extract dir already populated: {EXTRACT_DIR}")

    # Load training data (use full set like many spam baselines)
    X, y = load_training_data(EXTRACT_DIR)

    # Vectorizer + NB (spam-style)
    # IMPORTANT: keep these step names exactly like your spam model: "vectorizer", "classifier"
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z']+\b",
            max_features=200000,
        )),
        ("classifier", MultinomialNB())
    ])

    # Tune alpha like your spam script
    param_grid = {
        "classifier__alpha": [0.05, 0.1, 0.25, 0.5, 1.0]
    }

    print("[+] Training + tuning (GridSearchCV)...")
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    print("[+] Best params:", grid.best_params_)

    # Save model EXACTLY like spam solution (pipeline itself)
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"[+] Model saved to {MODEL_FILENAME}")

    # Upload model (exact pattern you provided)
    url = UPLOAD_URL
    model_file_path = MODEL_FILENAME

    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        response = requests.post(url, files=files)

    print("[+] Server response:")
    try:
        print(json.dumps(response.json(), indent=4))
    except Exception:
        print("[!] Server did not return JSON. Raw response:")
        print(response.status_code, response.text)


if __name__ == "__main__":
    main()

```

Below is the output from the code run above
```
[+] Starting spam-style sentiment pipeline
[+] Zip already exists: data/skills_assessment_data.zip
[+] Extract dir already populated: data/skills_assessment_data
[+] Loaded training data from: data/skills_assessment_data/train.json (samples=25,000)
[+] Training + tuning (GridSearchCV)...
Fitting 5 folds for each of 5 candidates, totalling 25 fits
[+] Best params: {'classifier__alpha': 1.0}
[+] Model saved to skills_assessment.joblib
[+] Server response:
{
    "accuracy": 1.0,
    "flag": "HTB{s3nt1m3nt_4n4lys1s_d4t4}"
}
```

