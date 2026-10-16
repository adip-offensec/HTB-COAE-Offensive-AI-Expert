

To evaluate your model, upload it to the evaluation portal running on the Playground VM. If you are not currently using the Playground VM, you can initialize it at the bottom of the page.

If you have the Playground VM running, you can use this Python script to upload your model from Jupyter directly. Once evaluated, if your model meets the required performance criteria, you will receive a flag value. This flag can be used to answer the question or verify the model’s success.

Code: python

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8000/api/upload"

# Path to the model file you want to upload
model_file_path = "spam_detection_model.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

If you are working from your own machine, ensure you have configured the HTB VPN to connect to the remote VM and spawned it. After connecting, access the model upload portal by navigating to `http://<VM-IP>:8000/` in your browser and then uploading your model.

1. What is the flag you get from submitting a good model for evaluation?
	1. Run the below code in JupiterLab

```python
import requests
import zipfile
import io
import os
import re
import json
import pandas as pd
import nltk
import joblib

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def main():
    # -------------------- Download dataset --------------------
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful")
    else:
        raise RuntimeError("Failed to download dataset")

    # -------------------- Extract dataset --------------------
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("sms_spam_collection")
        print("Extraction successful")

    # -------------------- Load dataset --------------------
    df = pd.read_csv(
        "sms_spam_collection/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
    )

    print(df.info())
    df = df.drop_duplicates()

    # -------------------- Preprocessing --------------------
    nltk.download("punkt")
    nltk.download("stopwords")

    df["message"] = df["message"].str.lower()
    df["message"] = df["message"].apply(
        lambda x: re.sub(r"[^a-zA-Z\s$!]", "", x)
    )
    df["message"] = df["message"].apply(word_tokenize)
    df["message"] = df["message"].apply(lambda x: " ".join(x))

    # -------------------- Vectorization --------------------
    vectorizer = CountVectorizer(stop_words="english")
    y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", MultinomialNB())
    ])

    # -------------------- Training --------------------
    param_grid = {"classifier__alpha": [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # -------------------- Test predictions --------------------
    new_messages = [
        "Congratulations! You have won a free lottery ticket!",
        "Hey, are we still meeting tomorrow?"
    ]

    predictions = best_model.predict(new_messages)
    probabilities = best_model.predict_proba(new_messages)

    for msg, pred, prob in zip(new_messages, predictions, probabilities):
        label = "Spam" if pred == 1 else "Ham"
        print(f"Message: {msg}")
        print(f"Prediction: {label}")
        print(f"Spam Probability: {prob[1]:.2f}")
        print(f"Ham Probability: {prob[0]:.2f}")
        print("-" * 50)

    # -------------------- Save model --------------------
    model_filename = "spam_detection_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")

    # -------------------- Upload model --------------------
    upload_url = "http://10.129.84.255:8000/api/upload"
    with open(model_filename, "rb") as model_file:
        files = {"model": model_file}
        upload_response = requests.post(upload_url, files=files)

    print(json.dumps(upload_response.json(), indent=4))


if __name__ == "__main__":
    main()
```

Below is the output of the above code

```
Download successful
Extraction successful
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   label    5572 non-null   object
 1   message  5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
None

[nltk_data] Downloading package punkt to /home//nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /home//nltk_data...
[nltk_data]   Package stopwords is already up-to-date!

Message: Congratulations! You have won a free lottery ticket!
Prediction: Spam
Spam Probability: 1.00
Ham Probability: 0.00
--------------------------------------------------
Message: Hey, are we still meeting tomorrow?
Prediction: Ham
Spam Probability: 0.00
Ham Probability: 1.00
--------------------------------------------------
Model saved to spam_detection_model.joblib
{
    "accuracy": 0.9130434782608695,
    "flag": "HTB{sp4m_cla55if13r_3v4lu4t0r}"
}
```

