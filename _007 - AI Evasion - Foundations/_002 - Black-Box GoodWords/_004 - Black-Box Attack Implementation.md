
The white-box attack had perfect vision: direct access to probability tables, goodness scores computed from internal parameters, optimal word selection from complete model knowledge. Now we operate blind. No probability tables. No internal parameters. Just a query interface that returns predictions. Can we still identify effective good words? Can we still achieve high evasion rates? The three-phase discovery algorithm answers both questions affirmatively.

## Discovery Setup and Budget Allocation

Operating under query constraints changes everything. We can't test every word exhaustively. We can't compute exact probability ratios. We must discover effective words through systematic experimentation within a fixed budget:

Code: python

```python
print("\n[*] Using three-phase discovery algorithm...")

# Build candidate vocabulary
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Built vocabulary of {len(candidate_words)} candidate words")
```

The vocabulary construction follows our hybrid approach from the Core Components section: frequency-based extraction from legitimate training messages plus manually curated conversational terms. This yields approximately 110-120 candidate words that strongly signal legitimate communication patterns. Without model access, we can't peek at which words the classifier learned as ham indicators. We must build our own candidate pool and empirically test which words actually work. Terms like "thanks", "tomorrow", "meeting", and informal contractions ("gonna", "wanna") become crucial - they're common in legitimate SMS communication but rare in spam.

The query budget determines our discovery capability. More queries mean better word discovery but higher cost and detection risk. Fewer queries reduce cost but might miss effective words:

Code: python

```python
# Show budget allocation
allocation = estimate_budget_allocation(query_budget)
print(f"\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f"    {phase:12}: {budget:4d} queries")
```

Code: txt

```txt
[*] Using three-phase discovery algorithm...
[+] Built vocabulary of 112 candidate words

[*] Budget allocation:
    exploration :  400 queries
    exploitation:  400 queries
    combination :  200 queries
```

For the default 1000-query budget, we allocate 400 queries each for exploration and exploitation (40% each), with 200 queries (20%) reserved for combination discovery. The vocabulary building identifies 112 candidate words from legitimate messages, providing a rich search space for our discovery algorithm.

## Running Three-Phase Discovery

Now we'll execute the complete three‑phase algorithm on the test spam messages. When evaluating messages, read the spam probability via `predict_proba(...)[0][1]` because the model returns probabilities as `[[p_ham, p_spam]]`:

Code: python

```python
# Run three-phase discovery
discovered_words, combination_scores, total_queries = three_phase_discovery(
    spam_test_messages[:50],
    candidate_words,
    budget=query_budget
)

print(f"\n[+] Discovery complete. Total queries used: {total_queries}/{query_budget}")
print(f"[+] Top 10 discovered words:")
for word, score in discovered_words[:10]:
    print(f"    {word:10} | impact: {score:.3f}")
```

Code: txt

```txt
[*] Phase 1: Exploration (budget: 400 queries)
  [P1 100/400] tested_words=10 | top3=need:0.166, say:0.005, thanks:0.000
  [P1 200/400] tested_words=17 | top3=happy:0.219, sleep:0.175, say:0.005
  [P1 300/400] tested_words=30 | top3=got:0.005, sleep:0.005, say:0.005
[+] Exploration complete. Queries: 400, Words tested: 40
  Top5 after exploration:
    happy        | score: 0.024
    sleep        | score: 0.001
    need         | score: 0.001
    got          | score: 0.001
    say          | score: 0.001

[*] Phase 2: Exploitation (budget: 400 queries)
  [P2 mid 200/400] top3=say:0.181, yeah:0.106, later:0.096
[+] Exploitation complete. Total queries: 800

[*] Phase 3: Combination search (budget: 200 queries)
  [P3 mid ~130/200] combos_tested=65 | best=happy:0.000
  [P3 mid ~200/200] combos_tested=130 | best=later + going:0.000
[+] Combination search complete. Total queries: 1000

[+] Discovery complete. Total queries used: 1000/1000
[+] Top discovered words (sample):
    really       | impact: 0.121
    got          | impact: 0.117
    eat          | impact: 0.115
    sleep        | impact: 0.078
    later        | impact: 0.068
```

The discovery operates on 50 spam messages from the test set, providing sufficient statistical diversity to identify generalizable patterns. Phase 1 (exploration) consumes 400 queries testing roughly 40 unique words, establishing initial effectiveness estimates across our candidate vocabulary. Each word gets tested on multiple messages to average out individual message variance. Phase 2 (exploitation) uses another 400 queries to refine these estimates through focused testing of the most promising candidates identified in phase 1. Phase 3 (combination search) explores word pairs and triplets with the remaining 200 queries, searching for synergistic combinations. The algorithm completes exactly at the 1000-query budget, respecting our constraint perfectly.

The results expose both similarities and differences compared to white-box discovery. In this run, the top discovered word "really" shows an impact of 0.121, meaning it reduces spam probability by 12.1 percentage points on average when added to test messages. Compare this to the white-box top performer "lor", which achieved only 4.0 percentage points of reduction. Why does black-box discovery sometimes find more impactful words?

The answer lies in how we measure impact. White-box goodness scores rank words by their probability ratio P(w∣ham)P(w∣spam), measuring theoretical ham-association from training data. Black-box discovery measures empirical impact: actual probability shifts when words are added to real spam messages and fed to the classifier. These metrics can disagree. A word might have a high goodness score (appears often in ham, rarely in spam) but low practical impact (doesn’t shift probabilities much when added to spam). Conversely, a word with moderate goodness might have high practical impact through semantic fit or reinforcing other features.

### Displaying Combinations

If combinations were discovered during Phase 3, display the most effective ones:

Code: python

```python
if combination_scores:
    print(f"\n[+] Top 5 word combinations:")
    top_combos = sorted(combination_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, score in top_combos:
        combo_str = ', '.join(combo)
        print(f"    {combo_str:30} | synergy: {score:.3f}")

# Update queries_used for compatibility
queries_used = total_queries
```

Word combinations often show enhanced effectiveness through synergy. Pairs like "later, going" can achieve greater impact than the sum of individual words. Triplets like "meeting, tomorrow, thanks" can reduce spam probability substantially beyond individual contributions. These synergistic effects occur when words semantically reinforce each other, creating stronger signals of legitimate communication.

## Testing Discovered Words

After discovery, we need to evaluate the discovered words on fresh spam messages to verify their effectiveness:

Code: python

```python
# Test discovered words
blackbox_results = []
test_counts = [0, 5, 10, 15, 20, 25, 30]  # Test with more words

for num_words in test_counts:
    if queries_used >= query_budget:
        break

    selected = [w for w, _ in discovered_words[:num_words]]
    evaded = 0
    tested = 0

    # Test on a different subset of spam messages
    eval_messages = spam_test_messages[30:50]  # Different messages from discovery
```

The evaluation tests different augmentation levels from 0 (baseline) to 30 words to understand the progression curve. We use messages 30-50 from the test set for evaluation, while discovery used 0-50. This partial overlap ensures fresh samples for unbiased evaluation while maintaining some consistency. The list comprehension extracts just the word strings, discarding their scores for the augmentation process.

### Measuring Evasion Success

For each configuration, we'll test augmented messages and track success rates:

Code: python

```python
    for msg in eval_messages:
        if queries_used >= query_budget:
            break

        aug = msg if num_words == 0 else msg + " " + " ".join(selected)
        vec = vectorizer.transform([aug])
        prob = classifier.predict_proba(vec)[0][1]  # spam prob
        queries_used += 1

        if prob < 0.5:  # evasion threshold
            evaded += 1
        tested += 1
```

After evaluating all messages for a given word count, we compute and record the evasion rate:

Code: python

```python
    if tested > 0:
        rate = (evaded / tested) * 100
        blackbox_results.append({'num_words': num_words, 'evasion_rate': rate})
        print(f"  Words: {num_words:2d} | Evasion: {rate:6.2f}% | Queries total: {queries_used}")

print(f"\n[+] Black-box attack complete. Total queries: {queries_used}/{query_budget}")
```

Each augmented message requires one query to check if it evades detection (spam probability < 0.5). The evasion rate calculation shows progressive improvement with more words. Typical results demonstrate the power of our discovered words: 0 words yields 5-10% evasion (natural misclassification), 10 words achieves 70-75% evasion, 15 words reaches 90-95% evasion, and 20+ words attains 95-100% evasion.

This progression, while slightly less steep than white-box results, still shows high effectiveness despite operating under severe information constraints.

## Comparison with White-Box Results

Compare the black‑box results with the white‑box attack to understand the trade‑offs:

Code: python

```python
print("\n" + "="*60)
print("ATTACK SUMMARY")
print("="*60)
print(f"Model Accuracy: {test_acc:.2%}")
print(f"Best White-box Evasion: {results_df['evasion_rate'].max():.1f}% @ {results_df.loc[results_df['evasion_rate'].idxmax(), 'num_words']} words")
if blackbox_results:
    bb_max = max(r['evasion_rate'] for r in blackbox_results)
    print(f"Best Black-box Evasion: {bb_max:.1f}% (with {queries_used} queries)")
print(f"Most Effective Word: '{word_impacts[0][0]}' (reduces spam prob by {word_impacts[0][1]:.1f}%)")
```

### Attack Comparison Visualization

![Line graph titled 'White-box vs Black-box Attack Comparison' showing evasion rate (%) increasing with the number of good words added. Compares white-box (full access) and black-box (840 queries) methods.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/attack_comparison.png)

The comparison typically reveals that query-constrained attacks achieve approximately 85-95% of the effectiveness of full-knowledge attacks. In our demonstration, the black-box discovery respected its 1000‑query budget while identifying highly effective words like "really" with 12.1% impact. This shows that empirical discovery through black‑box queries can identify different but equally powerful words compared to white‑box analysis.

### Black-Box Efficiency Analysis

What do our experiments reveal about black-box attack viability? Several findings emerge consistently across runs. The three-phase discovery algorithm successfully identifies high-impact words through adaptive exploration and exploitation, achieving 85-95% of white-box performance despite operating under severe information constraints. This isn't a marginal capability - it represents practical attack effectiveness sufficient for real-world deployment. An attacker with only query access can mount attacks nearly as effective as one with complete model knowledge.

Black-box discovery often finds different but equally effective words compared to white-box selection. Our run discovered "really" (12.1% impact) while white-box selected "lor" (4.0% impact). Both approaches achieve high overall evasion rates despite selecting different word sets. This demonstrates that multiple attack vectors exist - many different good word combinations can successfully evade the classifier. The attack surface is broad, not narrow.

What does this mean for defenders? Obscuring model internals provides limited security against adaptive attackers. Even without accessing probability tables or model parameters, attackers can empirically discover alternative attack vectors through systematic testing. The 1000-query budget proves sufficient - our algorithm completes within budget while identifying highly effective words. Disciplined allocation across exploration, exploitation, and combination search delivers strong results without exhaustive testing.

The attack succeeds because the fundamental vulnerability lies in the classifier's architecture, not in information leakage. Naive Bayes combines features additively in log space, creating systematic exploitability regardless of whether attackers know the exact probabilities. Black-box discovery takes longer and uses more queries than white-box extraction, but it reaches the same destination: high evasion rates through good word injection.

## Results Persistence

Finally, save all experimental results for reproducibility and further analysis:

Code: python

```python
# Save results
results = {
    'white_box': attack_results,
    'black_box': blackbox_results,
    'top_good_words': [(w, float(s)) for w, s, _, _ in top_good_words[:20]],
    'word_impacts': word_impacts[:10]
}

with open(output_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[+] Results saved to {output_dir / 'results.json'}")
print("[+] Attack demonstration complete!")
```

The results dictionary preserves all attack data in JSON format for later analysis. The list comprehension converts NumPy types to standard Python types for JSON compatibility - NumPy floats become Python floats, ensuring the file can be read by any JSON parser. This comprehensive output enables further analysis, comparison of attack strategies, and reproduction of our results.

