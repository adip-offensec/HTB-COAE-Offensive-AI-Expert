
We can employ adaptive learning techniques to efficiently discover effective words within query constraints. To this end, we'll implement epsilon-greedy selection and exponential moving averages to balance exploration of new candidates with exploitation of proven performers.

## Adaptive Scoring Functions

Build a scoring mechanism that maintains dynamic rankings of word effectiveness. We'll track both performance and testing frequency to make intelligent selection decisions:

### Initializing the Scorer

We'll create specialized data structures that track word performance across multiple dimensions:

Code: python

```python
def initialize_adaptive_scorer():
    """Initialize adaptive scoring data structures"""
    return {
        'word_scores': {},      # Maps word -> effectiveness score
        'word_counts': {},      # Maps word -> number of times tested
        'exploration_rate': 0.2  # 20% exploration for discovery phase
    }
```

The scorer maintains two separate mappings for different metrics. The `word_scores` dictionary stores effectiveness scores representing how much each word reduces spam probability - if "thanks" reduces spam probability by 0.15 on average, that's its score. The `word_counts` tracks testing frequency to identify under-tested words that might have unreliable scores. A word tested only once might have gotten lucky or unlucky, while one tested 50 times has a reliable average. The exploration rate of 0.2 means 20% of selections will explore new or rarely-tested words rather than exploiting known good ones.

## Epsilon-Greedy Word Selection

Now let's implement the epsilon-greedy algorithm, a standard approach from multi-armed bandit theory. This algorithm balances the exploration‑exploitation tradeoff by occasionally taking risks to discover better options:

Code: python

```python
def epsilon_greedy_select(scorer, available_words):
    """Select word using epsilon-greedy strategy

    Parameters:
        scorer (dict): Adaptive scorer state
        available_words (list): Candidate words to choose from

    Returns:
        str: Selected word for testing
    """
    import random

    if random.random() < scorer['exploration_rate']:
        # Exploration: try untested or rarely tested words
        untested = [w for w in available_words if w not in scorer['word_counts']]
        if untested:
            return random.choice(untested)
        else:
            # Choose least tested word
            return min(available_words,
                      key=lambda w: scorer['word_counts'].get(w, 0))
```

The `random.random()` generates a uniform value between 0.0 and 1.0. When this value falls below the exploration rate (20% of the time), the algorithm enters exploration mode. The list comprehension identifies words never tested by checking absence from `word_counts`. If untested words exist, `random.choice(untested)` selects one uniformly at random - every untested word has equal chance of being chosen.

When all words have been tested at least once, the `min()` function finds the least-tested word using the count as the key. If "meeting" was tested 3 times and "tomorrow" 15 times, we'd select "meeting" to balance our knowledge. This ensures we don't miss potentially good words just because they performed poorly in a single unlucky test.

### Exploitation Strategy

We now add the exploitation branch that always selects the current best‑scoring word to maximize immediate impact:

Code: python

```python
    else:
        # Exploitation: choose best performing word
        return max(available_words,
                  key=lambda w: scorer['word_scores'].get(w, 0))
```

In exploitation mode (80% of selections), the algorithm uses `max()` to find the highest-scoring word. The key function retrieves each word's effectiveness score, defaulting to 0 for untested words. If "got" has a score of 0.22 and "like" has 0.12, the algorithm selects "got" to maximize attack effectiveness. This greedy selection ensures most queries contribute to actual evasion rather than exploration.

The constant epsilon strategy is simple and effective in practice, but it does not provide logarithmic regret guarantees. With a fixed exploration rate it can incur linear regret in the worst case. Logarithmic regret O(log⁡T) typically requires algorithms such as UCB under i.i.d., bounded-reward assumptions, or an `epsilon` schedule that decays over time. In this section we use a fixed `epsilon` for clarity and treat UCB or annealed `epsilon` as advanced options.

## Score Updates with Exponential Moving Average

Let's implement a function that adjusts word scores based on new observations while maintaining historical context:

Code: python

```python
def update_word_score(scorer, word, impact, alpha=0.3):
    """Update word score using exponential moving average

    Parameters:
        scorer (dict): Adaptive scorer state
        word (str): Word being scored
        impact (float): Observed reduction in spam probability
        alpha (float): Learning rate
    """
    if word not in scorer['word_scores']:
        scorer['word_scores'][word] = impact
        scorer['word_counts'][word] = 1
    else:
        # Exponential moving average
        old_score = scorer['word_scores'][word]
        scorer['word_scores'][word] = (1 - alpha) * old_score + alpha * impact
        scorer['word_counts'][word] += 1
```

The `alpha` parameter of 0.3 controls adaptation speed, new observations receive 30% weight while historical data retains 70%. This balance prevents single outlier results from sharply changing our beliefs while still allowing adaptation.

When testing a word for the first time, its initial score equals the observed impact. For previously tested words, the exponential moving average formula `(1 - alpha) * old_score + alpha * impact` smoothly updates beliefs. If "thats" had score 0.10 and shows impact 0.15 in a new test, the updated score becomes (0.7 × 0.10) + (0.3 × 0.15) = 0.115. This gradual adjustment prevents overreaction to individual tests while tracking performance trends.

## Combinatorial Search for Synergies

Beyond individual word testing, we need to discover synergistic combinations where words amplify each other's effectiveness. Implement systematic combination testing:

### Systematic Combination Testing

We'll test word interactions to find super-additive effects:

Code: python

```python
def discover_word_combinations(message, test_words, max_size=3):
    """Discover effective word combinations through systematic search

    Parameters:
        message (str): Target spam message
        test_words (list): Promising words to test
        max_size (int): Maximum combination size

    Returns:
        dict: Mapping of word combinations to effectiveness scores
    """
    from itertools import combinations

    combination_scores = {}
    message_vec = vectorizer.transform([message])
    message_score = classifier.predict_proba(message_vec)[0][1]
```

The `combinations` function from `itertools` generates all possible k-element subsets from our word list. For 20 words and max_size=3, this creates 20 single words, 190 pairs, and 1140 triplets. The baseline `message_score` captures spam probability before modifications, typically 0.90-0.99 for actual spam messages.

### Testing Individual Words First

Before exploring combinations, we need to establish individual word effectiveness as a baseline:

Code: python

```python
    # Test individual words first
    for word in test_words[:20]:
        test_message = message + " " + word
        test_vec = vectorizer.transform([test_message])
        score = classifier.predict_proba(test_vec)[0][1]
        impact = message_score - score
        combination_scores[(word,)] = impact
```

Testing the top 20 words individually provides baseline measurements for synergy detection. The single-element tuple `(word,)` as the key maintains consistency with multi-word combinations - this allows us to treat all results uniformly regardless of combination size. The impact calculation quantifies effectiveness: if the original score was 0.95 and adding "got" reduces it to 0.73, the impact is 0.22.

### Discovering Pairwise Synergy

Now let's test word pairs to identify combinations with enhanced effects:

Code: python

```python
    # Test pairs for synergy
    if max_size >= 2:
        for word1, word2 in combinations(test_words[:15], 2):
            test_message = message + " " + word1 + " " + word2
            test_vec = vectorizer.transform([test_message])
            score = classifier.predict_proba(test_vec)[0][1]

            # Calculate synergy
            individual_impact = combination_scores.get((word1,), 0) + combination_scores.get((word2,), 0)
            actual_impact = message_score - score
            synergy = actual_impact - individual_impact

            if synergy > 0:  # Positive synergy detected
                combination_scores[(word1, word2)] = actual_impact
```

Testing pairs from the top 15 words generates 105 unique combinations (15 choose 2). The synergy calculation compares actual combined impact against the sum of individual impacts. If "got" reduces spam by 0.22 and "like" by 0.12, we expect 0.34 reduction together under independence. If the actual reduction is 0.38, the synergy of 0.04 indicates the pair works better together than separately.

This super-additive effect often occurs with semantically related words. For instance, "meeting" and "tomorrow" together strongly suggest legitimate scheduling communication, more than either word alone suggests legitimacy.

### Building Triplet Combinations

For the best‑performing pairs, explore three‑word combinations:

Code: python

```python
    # Test triplets for top pairs
    if max_size >= 3:
        top_pairs = sorted(
            [(k, v) for k, v in combination_scores.items() if len(k) == 2],
            key=lambda x: x[1], reverse=True
        )[:5]

        for pair, pair_score in top_pairs:
            for word in test_words[:10]:
                if word not in pair:
                    triplet = tuple(sorted(pair + (word,)))
                    test_message = message + " " + " ".join(triplet)
                    test_vec = vectorizer.transform([test_message])
                    score = classifier.predict_proba(test_vec)[0][1]
                    combination_scores[triplet] = message_score - score

    return combination_scores
```

The triplet search focuses on the 5 best pairs to limit computational cost. For each top pair, we try adding each of the top 10 individual words. The check `if word not in pair` prevents duplicates like ("got", "like", "got"). Sorting the triplet with `tuple(sorted(pair + (word,)))` ensures consistent storage regardless of word order - ("meeting", "report", "schedule") and ("schedule", "meeting", "report") are stored identically.

This focused search discovers powerful combinations that strongly signal legitimate communication. Triplets like ("meeting", "tomorrow", "thanks") can reduce spam probability by 40-50%, far exceeding the sum of individual effects.

