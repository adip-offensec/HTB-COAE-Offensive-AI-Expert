
The overall effectiveness curve tells us the attack works. But which words carry the most weight? How do probability distributions shift as we add more words? These deeper analyses reveal the attack's internal mechanisms and expose which configurations deliver maximum impact with minimal word addition.

## Word Impact Analysis

Not all good words are created equal. Some reduce spam probability dramatically when added alone, while others contribute only marginal shifts. Isolating each word's individual impact separates the heavy hitters from the supporting cast.

### Testing Individual Words

The experiment is straightforward: add each word individually to spam messages, measure the probability shift, average across all messages. Testing on 50 messages provides statistical reliability without excessive computation:

Code: python

```python
print("\n[*] Analyzing individual word impact...")

# Use 50 spam messages as a representative sample
sample_spam = spam_test_messages[:50]
word_impacts = []

print(f"[+] Testing {len(top_good_words[:20])} words on {len(sample_spam)} spam messages")
```

For each word, we'll measure its solo performance by comparing probabilities before and after augmentation:

Code: python

```python
for word, _, _, _ in top_good_words[:20]:
    total_impact = 0

    for message in sample_spam:
        # Calculate original spam probability
        vec_orig = vectorizer.transform([message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]  # spam prob

        # Calculate probability after adding the word
        vec_aug = vectorizer.transform([message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        # Measure the probability reduction
        impact = prob_orig - prob_aug
        total_impact += impact
```

The classifier's `predict_proba()` method returns probabilities as a 2D array with shape `[[p_ham, p_spam]]`. Index `[0][1]` extracts the spam probability from the first (and only) prediction. The baseline `prob_orig` typically exceeds 0.95 for genuine spam messages - the classifier is very confident before our intervention. After appending the word, `prob_aug` reveals the shifted probability. The difference `prob_orig - prob_aug` quantifies impact in absolute terms. A shift from 0.95 to 0.85 represents a 10 percentage point reduction, not a 10% relative reduction (which would only drop to 0.855).

Calculate the average impact and store results:

Code: python

```python
    # Calculate average impact across all messages
    avg_impact = (total_impact / len(sample_spam)) * 100
    word_impacts.append((word, avg_impact))

    # Show progress for first few words
    if len(word_impacts) <= 3:
        print(f"  Word '{word}': {avg_impact:.2f}% reduction")
```

We multiply by 100 to convert to percentage points for easier interpretation. This averaging approach accounts for variance - some words might work better on certain types of spam messages than others.

### Ranking Words by Effectiveness

Now sort words by their average impact to identify the most powerful ones:

Code: python

```python
word_impacts.sort(key=lambda x: x[1], reverse=True)
```

Code: txt

```txt
[*] Analyzing individual word impact...
[+] Testing 20 words on 50 spam messages
  Word 'lor': 3.98% reduction
  Word 'ü': 3.93% reduction
  Word '...': 3.89% reduction
Most Effective Word: 'lor' (reduces spam prob by 4.0%)
```

The top performer "lor" reduces spam probability by an average of 4.0 percentage points when added alone. This might seem modest - a single word only shifts probability by 4%. But context matters. The cumulative effect of combining multiple such words creates the 96%+ evasion rates we observed earlier. Ten words each contributing 4% would theoretically reduce spam probability by 40 percentage points, though in practice the effects aren't perfectly additive due to probability normalization constraints.

Notice the discrepancy between goodness scores and practical impact. The word "lor" topped our goodness rankings with a score of 50.22, indicating it appears 50× more frequently in ham than spam. Yet its practical impact averages only 4% when added to spam messages. Why the gap? Goodness scores measure theoretical ham-likelihood in isolation, while practical impact reflects the word's actual ability to shift already-spam-laden messages toward the ham classification. A word can be strongly associated with ham without having enough influence to overcome multiple strong spam indicators when added to a genuine spam message.

## Word Impact Visualization

Let's create a horizontal bar chart to see how individual words contribute to attack success:

### Creating the Bar Chart

We'll display word impacts as horizontal bars for easy comparison:

Code: python

```python
# Plot word impacts
fig, ax = plt.subplots(figsize=(10, 8), facecolor=NODE_BLACK)

words = [w for w, _ in word_impacts]
impacts = [i for _, i in word_impacts]
colors = [HTB_GREEN if i > 15 else NUGGET_YELLOW if i > 10 else HACKER_GREY for i in impacts]

bars = ax.barh(range(len(words)), impacts, color=colors, edgecolor='white', linewidth=0.5)
```

The figure dimensions (10×8) provide adequate height to display 20 words without crowding. The list comprehensions extract words and impacts from our tuples separately for plotting.

The `barh()` function creates horizontal bars, which work better than vertical bars for displaying word labels readably. White edges with 0.5 linewidth provide subtle separation between bars without overwhelming the visualization.

### Labeling and Styling

Now add clear labels and apply consistent styling:

Code: python

```python
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Average Spam Probability Reduction (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('Individual Word Impact on Spam Detection', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(axis='x', alpha=0.2)
ax.set_facecolor(NODE_BLACK)

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "word_impact.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'word_impact.png'}")
```

![Bar chart titled 'Individual Word Impact on Spam Detection' showing words like 'going,' 'gonna,' and 'lor' with average spam probability reduction percentages.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/word_impact.png)

The visualization exposes the distributed nature of attack effectiveness. No single word dominates. The highest-performing word `lor` reduces spam probability by approximately 4%, while most others contribute between 2% and 4%. The uniform grey coloring reflects that all tested words fall well below any threshold where individual words could single-handedly flip classifications. This isn't a weakness of our word selection - it's the mathematical reality of attacking Naive Bayes classifiers.

Why don't we see any "silver bullet" words that drastically shift probabilities alone? The classifier learns from thousands of messages during training. Strong spam indicators like "FREE", "WINNER", and currency symbols accumulate substantial probability mass favoring the spam class. A single good word, no matter how strongly associated with ham, cannot overcome this accumulated evidence. The visualization confirms what the mathematics predicted: successful evasion requires orchestrated combination of multiple good words.

This distributed impact pattern explains the 15-20 word threshold we observed for reliable evasion. Each word contributes a small but consistent probability shift. Five words reduce spam probability by roughly 15-20 percentage points. Ten words achieve 30-40 percentage points. Fifteen words push most messages across the decision boundary. The attack succeeds through accumulation, not individual word strength.

## Probability Shift Analysis

Now let's examine how spam probabilities change as we progressively add more good words. This analysis shows the attack's effect on individual messages:

### Sampling Messages for Analysis

We'll select representative messages and test them with increasing augmentation levels:

Code: python

```python
print("\n[*] Visualizing probability shifts...")

# Sample messages for detailed analysis
sample_messages = spam_test_messages[:8]
test_word_counts = [0, 5, 10, 20, 30]

fig, ax = plt.subplots(figsize=(14, 6), facecolor=NODE_BLACK)

x = np.arange(len(sample_messages))
width = 0.15
colors_list = [MALWARE_RED, NUGGET_YELLOW, AZURE, HTB_GREEN, AQUAMARINE]
```

Eight messages provide enough examples to see patterns without overcrowding the visualization. The test points (0, 5, 10, 20, 30 words) show key stages of the attack progression, baseline, initial impact, decision threshold, saturation, and beyond. The bar width of 0.15 allows five groups of bars to fit comfortably for each message. The color progression from `MALWARE_RED` (no augmentation) through to `AQUAMARINE` (maximum augmentation) visually represents the transformation from spam to ham classification.

### Creating Grouped Bar Chart

We need to calculate spam probabilities for each message at different augmentation levels. Iterate through the word counts and create grouped bars:

Code: python

```python
for i, num_words in enumerate(test_word_counts):
    # Select the appropriate number of good words
    selected = [w for w, _, _, _ in top_good_words[:num_words]]
    probs = []
```

Now augment each sample message and calculate its spam probability:

Code: python

```python
    for msg in sample_messages:
        # Augment message with selected words
        if num_words > 0:
            aug_msg = msg + " " + " ".join(selected)
        else:
            aug_msg = msg

        # Calculate spam probability
        vec = vectorizer.transform([aug_msg])
        spam_prob = classifier.predict_proba(vec)[0][1]
        probs.append(spam_prob)
```

Let's create the bars for this word count configuration:

Code: python

```python
    # Create grouped bars with distinct colors
    bars = ax.bar(x + i*width, probs, width,
                   label=f'{num_words} words',
                   color=colors_list[i], alpha=0.8)
```

Each iteration creates one set of bars for a specific word count. The x-position `x + i*width` shifts each group slightly to create the grouped effect. For message index 3, the bars appear at positions 3.0, 3.15, 3.30, 3.45, and 3.60. The 80% opacity (`alpha=0.8`) adds visual depth while maintaining clarity.

### Marking Successful Evasions

We'll highlight when messages successfully evade detection by crossing the decision boundary:

Code: python

```python
    # Mark successful evasions
    for j, (bar, prob) in enumerate(zip(bars, probs)):
        if prob < 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                   '✓', ha='center', va='bottom', color=HTB_GREEN, fontweight='bold')
```

The checkmark appears above any bar where spam probability drops below 0.5 - the classification threshold. The positioning `bar.get_x() + bar.get_width()/2` centers the mark horizontally on the bar, while `prob + 0.02` places it slightly above. The green checkmark provides immediate visual feedback about successful evasions, making it clear which configurations achieve our goal.

### Adding Decision Boundary

To orient the eye, let's add a reference line showing the decision threshold:

Code: python

```python
ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Decision boundary')
ax.set_xlabel('Message Index', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Spam Probability', fontsize=12, color=HTB_GREEN)
ax.set_title('Probability Shift with Increasing Good Words', fontsize=14, color=HTB_GREEN, pad=20)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'M{i+1}' for i in range(len(sample_messages))])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.2)
ax.set_facecolor(NODE_BLACK)
```

The decision boundary at 0.5 shows the threshold, probabilities above this line result in spam classification, below in ham classification. The x-tick positioning `x + width * 2` centers labels under each message's group of bars. Message labels like "M1", "M2" keep the x-axis clean while maintaining clear identification.

### Saving the Visualization

Finally, we'll apply consistent styling and save the completed visualization:

Code: python

```python
for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "probability_shift.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'probability_shift.png'}")
```

![Bar chart titled 'Probability Shift with Increasing Good Words' showing spam probability for messages M1 to M10 with varying numbers of good words. Includes a decision boundary line.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/probability_shift2.png)

The visualization shows attack effectiveness across eight spam messages from the test set. Two messages, M7 and M8, start with spam probabilities near 0.2 and 0.0 respectively, already below the 0.5 decision threshold at baseline. These are false negatives - spam messages the classifier already misclassifies as ham without any attack. For these messages, the attack is unnecessary since the classifier already fails.

The remaining six messages demonstrate actual attack success across different thresholds. M5 drops below 0.5 with just 5 added words. M2, M3, and M4 require 10 words to cross the decision boundary. Messages M1 and M6 maintain high spam probabilities (near 1.0) until 20 words force them below the threshold.

Each added word contributes log probability mass to the ham score through log⁡P(w|ham). The checkmarks show when accumulated ham evidence crosses the decision boundary. For M1 and M6, the baseline spam indicators are strong enough that 20 good words are needed to overcome them. For M5, just 5 words suffice. The attack succeeds on all six messages that the classifier initially gets right, though the word count needed varies based on baseline spam strength.

