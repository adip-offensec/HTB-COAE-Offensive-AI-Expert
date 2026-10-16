
# GoodWords Attack Implementation

Now that we have our trained spam filter, we will implement the GoodWords attack. We'll extract words strongly associated with legitimate messages and use them to manipulate the classifier's probability calculations, forcing spam messages to evade detection.

## GoodWords Extraction (White-Box)

White-box access gives us a significant advantage. Instead of guessing which words might fool the classifier, we can peer directly into its learned probability distributions and cherry-pick the most potent candidates:

Code: python

```python
print("\n[*] Extracting GoodWords from model...")

# Get feature names and probabilities
feature_names = vectorizer.get_feature_names_out()
ham_log_probs = classifier.feature_log_prob_[0]  # Ham class
spam_log_probs = classifier.feature_log_prob_[1]  # Spam class
```

The `get_feature_names_out()` method exposes our 3000-word vocabulary. The `feature_log_prob_` attribute holds log⁡P(w∣c) for each word w and class c. Index `[0]` accesses ham probabilities, `[1]` accesses spam probabilities. Why log space? Multiplying 3000 tiny probabilities (like 0.0013000) underflows to zero around 10−308. Working with their logarithms transforms multiplication into addition, keeping values numerically stable.

### Calculating Goodness Scores

Which words maximize our attack effectiveness? We need a scoring function that identifies words strongly associated with legitimate messages while being rare in spam. The goodness ratio provides this ranking by comparing conditional probabilities:

Code: python

```python
# Calculate goodness scores
goodness_scores = []
for i, word in enumerate(feature_names):
    ham_prob = np.exp(ham_log_probs[i])
    spam_prob = np.exp(spam_log_probs[i])
    goodness = ham_prob / (spam_prob + 1e-10)
    goodness_scores.append((word, goodness, ham_prob, spam_prob))
```

The exponential operation handles numerical precision carefully. When `ham_log_probs[i] = -5.3`, we get `np.exp(-5.3) ≈ 0.005`, recovering the original probability stored in log space. The epsilon value `1e-10` serves dual purposes: preventing division by zero when words never appear in spam, and ensuring goodness scores remain finite even for vocabulary terms absent from the spam training set. Without this safeguard, a word appearing 100 times in ham but zero times in spam would produce `goodness = 0.003 / 0.000`, triggering undefined behavior. With epsilon, we get `goodness = 0.003 / 0.00000000001 ≈ 300000000`, an astronomically high but numerically stable score.

The tuple structure `(word, goodness, ham_prob, spam_prob)` preserves complete information for analysis. We store both the derived goodness score and the underlying probabilities because later validation might require examining the raw conditional distributions. A word with goodness 50 could result from `ham_prob=0.005, spam_prob=0.0001` or from `ham_prob=0.5, spam_prob=0.01`. Both ratios equal 50, but the high-probability variant affects more messages.

Conversational terms dominate the top ranks. Words like "meeting", "tomorrow", and "thanks" score above 20. Spam indicators? They sink below 1. "FREE" scores 0.3. "WINNER" scores 0.2. "Claim" barely registers at 0.1.

### Selecting Top Good Words

Sort words by goodness scores to identify the most powerful words for evading detection:

Code: python

```python
# Sort by goodness
goodness_scores.sort(key=lambda x: x[1], reverse=True)
top_good_words = goodness_scores[:100]

print(f"[+] Top 10 GoodWords (most 'ham-like'):")
for word, score, hp, sp in top_good_words[:10]:
    print(f"    {word:15} | goodness: {score:8.2f} | ham_p: {hp:.4f} | spam_p: {sp:.4f}")
```

Code: txt

```txt
[*] Extracting GoodWords from model...
[+] Top 10 GoodWords (most 'ham-like'):
    lor             | goodness:    50.22 | ham_p: 0.0047 | spam_p: 0.0001
    ü               | goodness:    47.99 | ham_p: 0.0045 | spam_p: 0.0001
    ...             | goodness:    45.65 | ham_p: 0.0299 | spam_p: 0.0007
    da              | goodness:    45.39 | ham_p: 0.0042 | spam_p: 0.0001
    later           | goodness:    29.39 | ham_p: 0.0027 | spam_p: 0.0001
    doing           | goodness:    25.67 | ham_p: 0.0024 | spam_p: 0.0001
    ask             | goodness:    25.30 | ham_p: 0.0024 | spam_p: 0.0001
    really          | goodness:    24.55 | ham_p: 0.0023 | spam_p: 0.0001
    cos             | goodness:    23.81 | ham_p: 0.0022 | spam_p: 0.0001
    lol             | goodness:    23.81 | ham_p: 0.0022 | spam_p: 0.0001
```

The results reveal interesting linguistic patterns. "lor" (Singaporean slang) tops the list with a score of 50.22, appearing 50× more often in ham than spam. The ellipsis "..." scores 45.65, signaling conversational continuity. Temporal words ("later") and action verbs ("doing", "ask") all score above 20, reflecting the interactive nature of legitimate messages.

## Attack Implementation

How many good words does it take to fool the filter? One word? Ten? Twenty? We need empirical data. Testing different augmentation levels will reveal the attack's effectiveness curve and pinpoint the decision threshold.

### Extracting Spam Test Messages

The test set contains both ham and spam. We only care about spam for this experiment:

Code: python

```python
print("\n[*] Testing GoodWords attack...")

# Extract only spam messages for testing
spam_test_messages = X_test[y_test == 'spam']
print(f"[+] Testing on {len(spam_test_messages)} spam messages")
```

Code: txt

```txt
[*] Testing GoodWords attack...
[+] Testing on 128 spam messages
```

Our test set yields 128 spam samples, representing roughly 12% of the total test data. This sample size provides statistical reliability for measuring attack effectiveness while remaining computationally manageable. Each message will serve as a distinct test case, revealing how the attack performs across varied spam characteristics from promotional offers to phishing attempts.

### Setting Up the Experiment

Does evasion follow a linear progression? Does each added word contribute equally? Or does effectiveness follow some nonlinear curve with critical thresholds? Our mathematical analysis predicts a sigmoid pattern driven by the additive log-probability structure of Naive Bayes. Testing across multiple augmentation levels will reveal whether reality matches theory:

Code: python

```python
# Define test points from baseline (0) to saturation (40)
word_counts = [0, 5, 10, 15, 20, 25, 30, 35, 40]
attack_results = []

print(f"[*] Testing word counts: {word_counts}")
```

The test points balance coverage against computational cost. Zero words establishes baseline misclassification rate, typically 5-10% for well-trained models representing genuine false negatives. Five-word increments capture sufficient resolution to observe inflection points without redundancy. If the curve transitions sharply between 10 and 15 words, our granularity reveals this. If we tested every single word count, we'd waste 25 additional experiments (11, 12, 13, 14 words) that provide minimal new information. The upper bound of 40 words ensures we reach saturation where additional words provide zero marginal benefit. Early experiments on similar classifiers showed effectiveness plateaus around 20-25 words, so testing through 40 confirms we've captured the complete attack envelope.

### Implementing the Attack Loop

We'll systematically test each configuration against all spam messages. Why test every message at every word count? Individual spam characteristics vary dramatically. Some messages might evade with just 5 words due to weak spam signals, while others require 20+ words to overcome strong indicators like "FREE!!!" and currency symbols:

Code: python

```python
for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Show which words we're using (first iteration only for clarity)
    if num_words == 5:
        print(f"  Using words: {', '.join(selected_words)}")
```

Word selection uses the pre-sorted goodness rankings, ensuring we always add the most effective words first. This greedy approach isn’t globally optimal (word combinations might show synergy), but it’s computationally efficient and empirically effective. Testing every possible combination of 20 words from our top 100 would require evaluating (10020)≈5.4×1020 combinations. The greedy strategy completes in seconds.

### Message Augmentation Function

Let's create a small helper to handle the augmentation cleanly:

Code: python

```python
def augment_message(message, words_to_add):
    """Append good words to a message"""
    if len(words_to_add) > 0:
        return message + " " + " ".join(words_to_add)
    return message

# Test augmentation on one example using the top 5 words
sample_spam = spam_test_messages[0]
sample_augmented = augment_message(
    sample_spam,
    [w for w, _, _, _ in top_good_words[:5]]
)
print(f"\nOriginal: {sample_spam[:50]}...")
print(f"Augmented: {sample_augmented[:80]}...")
```

### Testing Evasion for Each Configuration

Now we'll test each spam message to see if it evades detection after augmentation:

Code: python

```python
for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Count how many spam messages evade after augmentation
    evaded = 0
    for message in spam_test_messages:
        # Augment the message
        augmented = augment_message(message, selected_words)

        # Transform and predict
        vec = vectorizer.transform([augmented])
        prob = classifier.predict_proba(vec)[0]

        # Check evasion: ham probability > spam probability
        if prob[0] > prob[1]:
            evaded += 1

    # Record results for this configuration
    evasion_rate = (evaded / len(spam_test_messages)) * 100
    attack_results.append({
        'num_words': num_words,
        'evasion_rate': evasion_rate,
        'evaded': evaded,
        'total': len(spam_test_messages)
    })

    print(f"  Words: {num_words:2d} | Evasion: {evasion_rate:6.2f}% ({evaded}/{len(spam_test_messages)})")
```

The classifier's decision boundary sits at probability 0.5. When `prob[0] > prob[1]`, the ham probability exceeds spam probability, triggering misclassification. This threshold proves critical: a message with `prob = [0.501, 0.499]` evades just as successfully as one with `prob = [0.99, 0.01]`, despite the dramatic confidence difference. The binary decision discards probability magnitude, caring only about which class wins. This creates an attack optimization target: we need only shift probabilities across 0.5, not drive them to extremes.

Code: python

```python
# Convert to DataFrame for easy plotting
results_df = pd.DataFrame(attack_results)
```

Code: txt

```txt
[*] Testing GoodWords attack...
[+] Testing on 128 spam messages
  Words:  0 | Evasion:   6.25% (8/128)
  Words:  5 | Evasion:  41.41% (53/128)
  Words: 10 | Evasion:  74.22% (95/128)
  Words: 15 | Evasion:  96.09% (123/128)
  Words: 20 | Evasion: 100.00% (128/128)
  Words: 25 | Evasion: 100.00% (128/128)
  Words: 30 | Evasion: 100.00% (128/128)
  Words: 35 | Evasion: 100.00% (128/128)
  Words: 40 | Evasion: 100.00% (128/128)
```

The results demonstrate exceptionally high attack effectiveness across all tested configurations. With zero words added (our baseline), only 6.25% of spam messages are misclassified as ham due to natural model errors. This represents 8 out of 128 messages - the classifier's inherent false negative rate. Adding just 5 carefully selected good words dramatically shifts this picture, increasing evasion to 41.41%. Nearly half of all spam messages (53 out of 128) now evade detection successfully.

The decision threshold emerges clearly at 10 words, where 74.22% of spam bypasses the filter. Three-quarters of our test messages (95 out of 128) successfully masquerade as legitimate. At 15 words, we achieve near-complete success with 96.09% evasion (123 out of 128 messages). By 20 words, the attack reaches 100% effectiveness. Every single spam message in our test set successfully evades the classifier.

What makes this progression particularly striking is its rapidity. Between 5 and 15 words, evasion explodes from 41.41% to 96.09%. This isn't gradual degradation - it's catastrophic failure. The bag-of-words model fundamentally cannot distinguish between genuine legitimate messages and spam messages augmented with good words. The additive nature of the probability calculations in log space means each additional word pushes the decision boundary further, and the classifier has no mechanism to detect the semantic mismatch between spam content and appended legitimate tokens. Feature manipulation wins decisively against this architecture.

## Attack Effectiveness Visualization

Raw numbers tell the story, but visualization reveals the pattern. The attack effectiveness should follow a characteristic sigmoid curve - the mathematical signature of Naive Bayes probability manipulation. Creating a plot will confirm this prediction and highlight the critical transition zone where the classifier's defenses collapse.

### Setting Up the Plot

Dark backgrounds provide better contrast for data visualization. We'll use the color scheme to maintain visual consistency with the rest of our materials:

Code: python

```python
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), facecolor=NODE_BLACK)

ax.plot(results_df['num_words'], results_df['evasion_rate'],
        marker='o', markersize=8, linewidth=2.5,
        color=HTB_GREEN, markeredgecolor='white', markeredgewidth=1)

ax.fill_between(results_df['num_words'], 0, results_df['evasion_rate'],
                alpha=0.3, color=HTB_GREEN)
```

The `'dark_background'` style provides a base dark theme that we customize with the HTB color palette defined earlier. The figure size of 12×6 inches creates a 2:1 aspect ratio, giving the horizontal progression curve adequate space to breathe without vertical compression. The `fill_between()` call with 30% opacity (`alpha=0.3`) adds a translucent area beneath the curve, emphasizing the cumulative nature of attack success as word count increases.

### Adding Reference Lines and Annotations

Threshold markers and annotations transform raw curves into decision-making tools. We'll mark the 50% and 90% evasion thresholds - the points where the attack transitions from experimental to operationally threatening:

Code: python

```python
# Add threshold lines
ax.axhline(y=50, color=NUGGET_YELLOW, linestyle='--', alpha=0.7, label='50% threshold')
ax.axhline(y=90, color=AZURE, linestyle='--', alpha=0.7, label='90% threshold')

# Highlight maximum
max_idx = results_df['evasion_rate'].idxmax()
max_rate = results_df.loc[max_idx, 'evasion_rate']
max_words = results_df.loc[max_idx, 'num_words']
ax.scatter(max_words, max_rate, s=200, color=MALWARE_RED, zorder=5)
ax.annotate(f'Peak: {max_rate:.1f}%\n@ {max_words} words',
           xy=(max_words, max_rate), xytext=(max_words+5, max_rate-10),
           color='white', fontsize=10,
           arrowprops=dict(arrowstyle='->', color=MALWARE_RED, lw=1.5))
```

The 50% threshold line in `NUGGET_YELLOW` represents the equipoise point where the attack becomes more successful than not - half of spam messages evade detection. Crossing this threshold means the filter fails more often than it succeeds for adversarially-augmented messages. The 90% threshold in `AZURE` indicates near-complete evasion, where only 10% of spam remains caught. This represents practical attack success for real-world deployment.

The `zorder=5` parameter controls rendering order, ensuring our scatter point marking the peak appears above other plot elements like grid lines or the fill area. The annotation positioning `xytext=(max_words+5, max_rate-10)` offsets the label 5 units right and 10 units down from the peak point, preventing visual overlap with the main curve while maintaining clear association through the arrow connector.

### Applying Final Styling

Now we'll apply consistent styling and save our visualization:

Code: python

```python
ax.set_xlabel('Number of Good Words Added', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Evasion Rate (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('GoodWords Attack Effectiveness', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(True, alpha=0.2)
ax.set_facecolor(NODE_BLACK)
ax.legend()

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
output_dir = Path("attachments")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "attack_effectiveness.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"\n[+] Plot saved to {output_dir / 'attack_effectiveness.png'}")
```

![Graph titled 'GoodWords Attack Effectiveness' showing evasion rate (%) increasing with the number of good words added, reaching a maximum of 100%. Includes 50% threshold and 90% success lines.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/attack_effectiveness.png)

The visualization confirms our mathematical prediction. The curve follows the characteristic sigmoid shape expected from Naive Bayes probability manipulation. Why this S-curve pattern? The mathematics demands it. In log space, each good word contributes additively to the ham class score:

log⁡P(ham∣message)=log⁡P(ham)+∑w∈wordslog⁡P(w∣ham)

This linear accumulation in log space transforms into a nonlinear sigmoid response when we convert back to probability space through the softmax function used for classification. The curve exposes three distinct attack phases.

Phase one extends from 0 to 5 words, where evasion remains low at 6.25

Phase three begins at 15 words and reaches saturation at 20 words with 100% evasion. The decision boundary has completely shifted. Adding more words beyond 20 provides no additional benefit because every message already evades detection. The classifier's probabilistic framework has been fully compromised.

