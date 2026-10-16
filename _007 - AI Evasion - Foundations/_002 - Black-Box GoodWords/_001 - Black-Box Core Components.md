
The black-box attack scenario simulates realistic conditions where attackers have only query access to the target model. Without access to internal parameters, we must discover effective words through systematic exploration within a limited query budget. Build the core components that enable this discovery process.

## Attack Simulation Setup

We'll begin by establishing constraints that mirror real-world API limitations. Most production systems impose rate limits or charge per prediction:

Code: python

```python
print("\n[*] Simulating black-box attack scenario...")
print("[*] Budget: 1000 queries")

# Simulate limited query access
query_budget = 1000
queries_used = 0
query_log = []
```

The query budget of 1000 represents typical rate limits or cost constraints in production systems. Many hosted NLP services charge per prediction or impose daily limits, so careful budgeting is necessary. The `queries_used` counter tracks our consumption throughout the attack, ensuring we don't exceed limits. The `query_log` list could store detailed timing and response data for analysis, though we won't use it in this basic implementation.

## Building Candidate Vocabulary

We need a `candidate vocabulary` that imitates legitimate language under black‑box constraints. The plan is to mine common tokens from `ham` messages, apply simple filters for reliability, then merge a small curated list that captures conversational style.

### Extracting Words from Ham Messages

First, we extract ham messages from the training set and build a frequency table. We analyze a bounded sample for efficiency, and we apply a light length filter to remove stopword-like tokens and artifacts while preserving informative conversational words. This step isolates language that authentically appears in legitimate communication, which is exactly what a black-box attacker tries to mimic.

Code: python

```python
def extract_ham_word_freq(X_train, y_train, sample_size=500):
    """
    Compute token frequencies from a sample of ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.

    Returns
    -------
    dict[str, int]
        Mapping of word -> frequency within sampled ham messages.
    """
    ham_msgs = X_train[y_train == 'ham']
    limit = min(sample_size, len(ham_msgs))
    freq = {}
    for msg in ham_msgs[:limit]:
        for w in str(msg).split():
            if 2 < len(w) < 10:  # keep typical conversational tokens
                freq[w] = freq.get(w, 0) + 1
    return freq

wf_example = extract_ham_word_freq(X_train, y_train, sample_size=500)
print("[*] Example: extract_ham_word_freq")
print(f"  Ham messages sampled: {min(500, sum(y_train == 'ham'))}")
print(f"  Unique tokens found: {len(wf_example)}")
top5 = sorted(wf_example.items(), key=lambda x: (-x[1], x[0]))[:5]
for w, c in top5:
    print(f"    {w}: {c}")
```

Code: txt

```txt
[*] Example: extract_ham_word_freq
  Ham messages sampled: 500
  Unique tokens found: 2076
    you: 154
    and: 104
    the: 102
    have: 57
    but: 46
```

This function mirrors how an attacker would gather data in the wild. It assumes access to some legitimate text, for example public forums or personal communications, and counts word usage. The `split()` method performs basic tokenization that matches our earlier preprocessing. The length filter `(3..9)` removes uninformative tokens while retaining short, conversational words that often indicate ham.

The length constraints `len(word) > 2 and len(word) < 10` filter out unhelpful tokens. Words with 2 or fewer characters (like "a", "I", "to") often appear in stop word lists and provide little discriminative value. Words longer than 10 characters are often URLs, phone numbers, or other artifacts rather than meaningful vocabulary. For instance, we'd exclude both "a" (too short) and "http://example.com" (too long), focusing on words like "meeting", "thanks", "tomorrow" that genuinely indicate legitimate communication.

### Selecting High-Frequency Words

Why prioritize frequency? Words appearing often in legitimate messages signal common communication patterns the classifier learned as ham indicators. Rare words might be flukes or dataset artifacts that don't generalize across different spam instances:

Code: python

```python
def select_high_frequency_words(word_freq, max_words=100, min_freq=5):
    """
    Select the most frequent ham words above a minimum frequency.

    Parameters
    ----------
    word_freq : dict[str, int]
        Token frequency table for sampled ham messages.
    max_words : int, default 100
        Maximum number of words to return.
    min_freq : int, default 5
        Minimum frequency a word must meet to be considered.

    Returns
    -------
    list[str]
        Top words sorted by decreasing frequency then lexicographically.
    """
    sorted_by_freq = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    top = [w for w, c in sorted_by_freq if c > min_freq][:max_words]
    return top

top_words_example = select_high_frequency_words(wf_example, max_words=100, min_freq=5)
print("[*] Example: select_high_frequency_words")
print(f"  Selected top words: {len(top_words_example)} (min_freq=5)")
print("  First 10:", ", ".join(top_words_example[:10]))
```

Code: txt

```txt
[*] Example: select_high_frequency_words
  Selected top words: 100 (min_freq=5)
  First 10: you, and, the, have, but, for, that, i'm, all, your
```

The tuple key `(-x[1], x[0])` implements a two-tier sort: primary by negative frequency (descending), secondary by word string (ascending). This lexicographic tie-breaking ensures deterministic ordering when words have identical frequencies. Without it, `sorted()` produces unstable results for ties, potentially returning different orderings across Python versions or runs. The slice `[:max_words]` caps computational cost downstream. Testing all 2076 unique tokens would require thousands of queries, but 100 candidates fit comfortably within a 1000-query budget while capturing the most promising ham signals.

### Merging Curated Conversational Terms

Beyond frequency-based selection, we include carefully chosen conversational words that often appear in legitimate messages. These capture informal style, temporal references, and polite terms that are underrepresented in spam. This curated set hedges against sampling bias when the ham sample is small and against distribution shift in a black-box setting where we cannot inspect the model’s internal vocabulary.

Code: python

```python
def merge_with_curated(top_words, additional_candidates=None):
    """
    Merge data-driven top words with curated conversational candidates.

    Parameters
    ----------
    top_words : list[str]
        High-frequency ham words from the previous step.
    additional_candidates : list[str] | None
        Optional curated list to include regardless of frequency.

    Returns
    -------
    list[str]
        Deduplicated merged list (lexicographically ordered).
    """
    if additional_candidates is None:
        additional_candidates = [
            "ok", "cos", "ill", "thats", "later", "said", "ask", "didnt",
            "dont", "doing", "going", "come", "home", "tomorrow", "today", "sorry",
            "thanks", "yeah", "yes", "sure", "see", "tell", "know", "think",
        ]
    merged = set(top_words) | set(additional_candidates)
    return sorted(merged)

merged_example = merge_with_curated(top_words_example)
added = sorted(set(merged_example) - set(top_words_example))
print("[*] Example: merge_with_curated")
print(f"  Merged size: {len(merged_example)} | Added curated: {len(added)}")
print("  Sample added terms:", ", ".join(added[:5]))
```

Code: txt

```txt
[*] Example: merge_with_curated
  Merged size: 112 | Added curated: 12
  Sample added terms: cos, didnt, doing, ill, later
```

These additional words represent common patterns in legitimate communication that rarely appear in spam. Informal contractions like `dont` and temporal terms like `tomorrow` signal conversational context. Deduplication ensures each candidate is unique before testing. The example prints show how many curated terms were actually added on your data.

### Assembling and Verifying

We now compose the previous steps into a single helper so we can reuse the exact process later on. The function extracts ham frequencies, selects high-frequency words, merges curated terms, and returns a stable, reproducible ordering. The final ordering is deterministic, primary sort by ham frequency and tie-break by lexicographic order. This makes results reproducible and easy to test.

Code: python

```python
def build_candidate_vocabulary(
    X_train,
    y_train,
    sample_size=500,
    max_words=100,
    min_freq=5,
    additional_candidates=None,
):
    """
    Build a candidate vocabulary for black-box discovery from ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.
    max_words : int, default 100
        Maximum number of top frequent ham words to keep before merging extras.
    min_freq : int, default 5
        Minimum frequency threshold for inclusion from the ham corpus.
    additional_candidates : list[str] | None
        Optional curated conversational terms to include.

    Returns
    -------
    list[str]
        Deduplicated candidate words ordered by decreasing ham frequency,
        then lexicographically for stable ties.
    """
    word_freq = extract_ham_word_freq(X_train, y_train, sample_size=sample_size)
    top_words = select_high_frequency_words(word_freq, max_words=max_words, min_freq=min_freq)
    merged = merge_with_curated(top_words, additional_candidates=additional_candidates)

    # Stable final ordering driven by ham frequency, then lexical for ties
    def sort_key(w):
        return (-word_freq.get(w, 0), w)

    return sorted(merged, key=sort_key)

cv_example = build_candidate_vocabulary(X_train, y_train)
print("[*] Example: build_candidate_vocabulary")
print(f"  Candidates: {len(cv_example)}")
print("  First 10:", ", ".join(cv_example[:10]))
```

Code: txt

```txt
[*] Example: build_candidate_vocabulary
  Candidates: 112
  First 10: you, and, the, have, but, for, that, i'm, all, your
```

### Building the Candidate List

Finally, we build the candidate list using the helper and display its size. This call produces the exact list that the discovery algorithms will use later on. The printed count offers a quick health check that your configuration matches the expected range.

Code: python

```python
# Build candidate vocabulary for discovery
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Testing {len(candidate_words)} candidate words extracted from ham messages")
```

Code: txt

```txt
[+] Testing 112 candidate words extracted from ham messages
```

## Query Management Framework

The black-box attack must carefully manage its query budget across discovery and exploitation phases. Establish the framework for this management:

### Budget Allocation Strategy

We'll reserve portions of the budget for different phases. First we implement a helper that returns a 40-40-20 split across `exploration`, `exploitation`, and `combination`, using the remainder for the last phase so totals always match the input. Then we print an example allocation for the current `query_budget`.

Code: python

```python
def estimate_budget_allocation(total_budget):
    """
    Estimate allocation across exploration, exploitation, and combination.

    Parameters
    ----------
    total_budget : int
        Total query budget available for discovery.

    Returns
    -------
    dict
        Mapping phase -> integer number of queries that sums to `total_budget`.
    """
    explore = int(0.4 * total_budget)
    exploit = int(0.4 * total_budget)
    combine = total_budget - explore - exploit  # absorb rounding
    return {
        'exploration': explore,
        'exploitation': exploit,
        'combination': combine,
    }

# Quick demo for budget allocation
allocation = estimate_budget_allocation(query_budget)
print("\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f"  {phase:12}: {budget:4d} queries")
print(f"  Total: {sum(allocation.values())} / {query_budget}")
```

Code: txt

```txt
[*] Budget allocation:
  exploration :  400 queries
  exploitation:  400 queries
  combination :  200 queries
  Total: 1000 / 1000
```

Now we prepare inputs for Phase 1 by selecting a diverse subset of spam messages and shuffling candidates to avoid ordering bias:

Code: python

```python
# Discovery phase - test word effectiveness
word_scores = {}
test_spam_samples = spam_test_messages[:50]  # More test messages

# Test in batches to be more efficient
print(f"[*] Discovery phase: testing {len(candidate_words)} candidates...")

# Randomly sample candidates and messages for better coverage
np.random.shuffle(candidate_words)
np.random.shuffle(test_spam_samples)
```

Using 50 test spam messages provides statistical reliability while limiting query consumption. If we only tested on 5 messages, random variance could mislead us about word effectiveness. The shuffling operations randomize the testing order, preventing systematic biases if the attack is interrupted or reaches budget limits early. For instance, without shuffling, we might test all frequency-based words first and never reach the additional words if we hit the budget limit.

The implementation uses a 40-40-20 allocation: `exploration` 40%, `exploitation` 40%, and `combination` 20% (as shown by the `estimate_budget_allocation` helper). When we refer to `discovery`, we mean `exploration + combination` (60% total). This balance offers broad search and targeted refinement within a fixed budget. In tighter budgets, the split can be adjusted, for example increasing `exploitation` when immediate evasion is the goal or increasing `exploration` when many queries are available.

