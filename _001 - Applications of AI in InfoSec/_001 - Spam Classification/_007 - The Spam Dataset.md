

We'll explore Bayesian spam classification using the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a curated resource tailored for developing and evaluating text-based spam filters. This dataset emerges from the combined efforts of Tiago A. Almeida and Akebo Yamakami at the School of Electrical and Computer Engineering at the University of Campinas in Brazil, and José María Gómez Hidalgo at the R&D Department of Optenet in Spain.

Their work, "`Contributions to the Study of SMS Spam Filtering: New Collection and Results`," presented at the 2011 ACM Symposium on Document Engineering, aimed to address the growing problem of unsolicited mobile phone messages, commonly known as `SMS spam`. Recognizing that many existing spam filtering resources focused on email rather than text messages, the authors assembled this dataset from multiple sources, including the Grumbletext website, the NUS SMS Corpus, and Caroline Tag’s PhD thesis.

The resulting corpus contains 5,574 text messages annotated as either `ham` (legitimate) or `spam` (unwanted), making it a great resource for building and testing models that can differentiate meaningful communications from intrusive or deceptive ones. In this context, `ham` refers to messages from known contacts, subscriptions, or newsletters that hold value for the recipient, while `spam` represents unsolicited content that typically offers no benefit and may even pose risks to the user.

## Downloading the Dataset

The first step in our process is to download this dataset, and we'll do it programmatically in our notebook.

Code: python

```python
import requests
import zipfile
import io

# URL of the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")
```

We use the `requests` library to send an HTTP GET request to the URL of the dataset. We check the status code of the response to determine if the download was successful (`status_code == 200`).

After downloading the dataset, we need to extract its contents. The dataset is provided in a `.zip` file format, which we will handle using Python's `zipfile` and `io` libraries.

Code: python

```python
# Extract the dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction successful")
```

Here, `response.content` contains the binary data of the downloaded `.zip` file. We use `io.BytesIO` to convert this binary data into a bytes-like object that can be processed by `zipfile.ZipFile`. The `extractall` method extracts all files from the archive into a specified directory, in this case, `sms_spam_collection`.

It's useful to verify that the extraction was successful and to see what files were extracted.

Code: python

```python
import os

# List the extracted files
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)
```

The `os.listdir` function lists all files and directories in the specified path, allowing us to confirm that the `SMSSpamCollection` file is present.

## Loading the Dataset

With the dataset extracted, we can now load it into a `pandas` DataFrame for further analysis. The SMS Spam Collection dataset is stored in a tab-separated values (TSV) file format, which we specify using the `sep` parameter in `pd.read_csv`.

Code: python

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)
```

Here, we specify that the file is tab-separated (`sep="\t"`), and since the file does not contain a header row, we set `header=None` and provide column names manually using the `names` parameter.

After loading the dataset, it is important to inspect it for basic information, missing values, and duplicates. This helps ensure that the data is clean and ready for analysis.

Code: python

```python
# Display basic information about the dataset
print("-------------------- HEAD --------------------")
print(df.head())
print("-------------------- DESCRIBE --------------------")
print(df.describe())
print("-------------------- INFO --------------------")
print(df.info())
```

To get an overview of the dataset, we can use the `head`, `describe`, and info `methods` provided by pandas.

- `df.head()` displays the first few rows of the DataFrame, giving us a quick look at the data.
- `df.describe()` provides a statistical summary of the numerical columns in the DataFrame. Although our dataset is primarily text-based, this can still be useful for understanding the distribution of labels.
- `df.info()` gives a concise summary of the DataFrame, including the number of non-null entries and the data types of each column.

Checking for missing values is crucial to ensure that our dataset does not contain any incomplete entries.

Code: python

```python
# Check for missing values
print("Missing values:\n", df.isnull().sum())
```

The `isnull` method returns a DataFrame of the same shape as the original, with boolean values indicating whether each entry is null. The `sum` method then counts the number of `True` values in each column, giving us the total number of missing entries.

Duplicate entries can skew the results of our analysis, so it's important to identify and remove them.

Code: python

```python
# Check for duplicates
print("Duplicate entries:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()
```

The `duplicated` method returns a boolean Series indicating whether each row is a duplicate or not. The `sum` method counts the number of `True` values, giving us the total number of duplicate entries. We then use the `drop_duplicates` method to remove these duplicates from the DataFrame.

