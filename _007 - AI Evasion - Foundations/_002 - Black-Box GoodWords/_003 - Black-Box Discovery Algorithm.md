
The complete black-box discovery process integrates vocabulary building, adaptive scoring, and combination search through a three-phase approach. We'll balance discovering new effective words with refining knowledge about proven performers, all while respecting strict query budget constraints.

## Three-Phase Discovery Function

Every query we make teaches us something. Testing `thanks` on a spam message reveals how effective that word is at reducing spam scores. But queries are precious resources, limited by budget constraints. The key question becomes: which queries teach us the most? Should we test new words to expand our knowledge, or should we retest promising words to refine our estimates?

The discovery algorithm allocates queries across exploration, exploitation, and combination phases to maximize effectiveness. From an information theory perspective, each query provides information gain:

I(Q)=H(W)−H(W∣Q)

This formula quantifies learning. Think of entropy $H$ as measuring our uncertainty. Before testing a word, we have high uncertainty about its effectiveness. After testing, we have less uncertainty. The information gain $I(Q)$ is the reduction in uncertainty that the query provides.

where H(W) is the entropy of word effectiveness before the query and H(W∣Q) is the entropy after observing the query result.

Imagine you know nothing about 100 candidate words. Your uncertainty is maximal because any word could be great or terrible. Test one word and observe it reduces spam probability by 0.15. Now you have less uncertainty about that word specifically, while the other 99 remain unknown. Your overall uncertainty has decreased, representing information gain from that query.

The three-phase approach maximizes cumulative information gain through targeted allocation:

- The exploration phase maximizes entropy reduction across the full vocabulary, casting a wide net to identify promising candidates. Testing many different words reveals the overall landscape of effectiveness.
- The exploitation phase refines estimates for high-value words where variance is highest, focusing queries where uncertainty matters most. We already know these words work, now we need accurate measurements of how well.
- The combination phase discovers non-linear synergies between words that can't be predicted from individual performance. Some word pairs achieve greater impact together than their individual scores would predict.

This allocation strategy approximates a good solution to the budgeted information maximization problem, balancing breadth of knowledge (testing many words) with depth of understanding (accurately measuring the best words).

### Algorithm Initialization

Let's implement the three‑phase discovery function by setting up tracking structures and dividing the budget:

Code: python

```python
def three_phase_discovery(spam_messages, candidate_words, budget=1000):
    """Three-phase discovery: exploration, exploitation, combination

    Parameters:
        spam_messages (list): Target spam messages
        candidate_words (list): Vocabulary to test
        budget (int): Total query budget

    Returns:
        tuple: (discovered_words, combination_scores, queries_used)
    """
    scorer = initialize_adaptive_scorer()
    queries_used = 0

    # Allocate budgets using 40-40-20 split strategy
    allocation = estimate_budget_allocation(budget)
    exploration_budget = allocation['exploration']
    exploitation_budget = allocation['exploitation']
    combination_budget = allocation['combination']
```

The function accepts spam messages to test against, candidate vocabulary extracted from legitimate messages, and a query budget (defaulting to 1000). The scorer initialization creates the adaptive scoring structures with word scores, counts, and a 20% exploration rate. The `queries_used` counter ensures we respect budget constraints throughout execution. The budget allocation uses the 40-40-20 split strategy detailed in the Core Components section, balancing broad search with targeted refinement.

## Phase 1: Broad Exploration

The goal in exploration is to test a broad set of candidate words against varied spam messages to estimate which words reduce spam probability. We read the model’s spam probability via `predict_proba(...)[0][1]` and define a word’s `impact` for a single test as the reduction in that probability when the word is appended to the message.

### Initialisation And Milestones

We announce the phase and create progress markers at 25%, 50%, and 75% of the exploration budget. The set `p1_reported` ensures each milestone prints once.

Code: python

```python
    # Phase 1: Broad exploration (allocated budget)
    print(f"
[*] Phase 1: Exploration (budget: {exploration_budget} queries)")

    p1_marks = {
        max(1, int(0.25 * exploration_budget)),
        max(1, int(0.50 * exploration_budget)),
        max(1, int(0.75 * exploration_budget)),
    }
    p1_reported = set()
```

### Selecting A Message And A Candidate Word

Each iteration samples one spam message uniformly. The `epsilon_greedy_select` chooses a candidate word, balancing exploration of untested words with exploitation of current high scorers in `scorer['word_scores']`.

Code: python

```python
    # Select a message and a candidate word
    test_message = random.choice(spam_messages)
    word = epsilon_greedy_select(scorer, candidate_words)
```

### Measuring Impact With Two Queries

We issue two queries per test, the baseline message and the augmented message with the candidate appended. The difference defines the per‑test `impact`:

Code: python

```python
    # Baseline and augmented spam probabilities
    vec_orig = vectorizer.transform([test_message])
    prob_orig = classifier.predict_proba(vec_orig)[0][1]  # spam prob

    vec_aug = vectorizer.transform([test_message + " " + word])
    prob_aug = classifier.predict_proba(vec_aug)[0][1]

    impact = prob_orig - prob_aug
```

### Updating Scores And Accounting For Budget

We update the running effectiveness estimate via exponential moving average and account for the two consumed queries.

Code: python

```python
    # Update running score and consume query budget
    update_word_score(scorer, word, impact)
    queries_used += 2
```

### Milestone Reporting

At milestones we show how many unique words have been tested and the current top three by score.

Code: python

```python
    # Optional milestone report
    if queries_used in p1_marks and queries_used not in p1_reported:
        top3 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(
            f"  [P1 {queries_used}/{exploration_budget}] "
            f"tested_words={len(scorer['word_scores'])} | "
            f"top3=" + ", ".join(f"{w}:{s:.3f}" for w, s in top3)
        )
        p1_reported.add(queries_used)
```

### Putting The Loop Together

We repeat selection, measurement, update, and optional milestone reporting until the exploration budget is consumed or we run out of candidates.

Code: python

```python
    while queries_used < exploration_budget and len(candidate_words) > 0:
        # Select inputs
        test_message = random.choice(spam_messages)
        word = epsilon_greedy_select(scorer, candidate_words)

        # Measure impact with two queries
        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]
        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]
        impact = prob_orig - prob_aug

        # Update score and account for budget
        update_word_score(scorer, word, impact)
        queries_used += 2

        # Milestone report
        if queries_used in p1_marks and queries_used not in p1_reported:
            top3 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(
                f"  [P1 {queries_used}/{exploration_budget}] "
                f"tested_words={len(scorer['word_scores'])} | "
                f"top3=" + ", ".join(f"{w}:{s:.3f}" for w, s in top3)
            )
            p1_reported.add(queries_used)
```

### Exploration Results

After exploration, we'll report our progress:

Code: python

```python
    print(f"[+] Exploration complete. Queries: {queries_used}, Words tested: {len(scorer['word_scores'])}")
    top5 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        print("  Top5 after exploration:")
        for w, s in top5:
            print(f"    {w:12} | score: {s:.3f}")
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
```

Typically, exploration tests 40-50 unique words within the 400-query budget. Some words get tested multiple times due to the epsilon-greedy selection's exploitation component. This provides initial effectiveness estimates across a broad vocabulary sample, identifying promising candidates for deeper investigation.

## Phase 2: Focused Exploitation

The exploitation phase refines our understanding of the most promising words within the allocated budget:

Code: python

```python
    # Phase 2: Focused exploitation
    scorer['exploration_rate'] = 0.1  # Reduce exploration

    # Get top words for exploitation
    top_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:30]
    top_word_list = [w for w, _ in top_words]

    print(f"\n[*] Phase 2: Exploitation (budget: {exploitation_budget} queries)")
    initial_queries = queries_used
    p2_mid = initial_queries + max(1, exploitation_budget // 2)
```

This phase focuses on refining estimates for the top 30 words discovered during exploration. Reducing the exploration rate to 0.1 shifts focus toward exploiting known good words. We've already identified promising candidates, now we need accurate estimates of their effectiveness. The sorting operation orders words by effectiveness score, from highest to lowest impact.

### Concentrated Testing

We repeatedly test top‑performing words against representative messages to refine impact estimates and reduce variance:

Code: python

```python
    while queries_used < initial_queries + exploitation_budget and len(top_word_list) > 0:
        test_message = random.choice(spam_messages[:20])  # Focus on fewer messages
        word = random.choice(top_word_list[:15])  # Focus on best words

        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]

        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        impact = prob_orig - prob_aug
        update_word_score(scorer, word, impact)
        queries_used += 2

    print(f"[+] Exploitation complete. Total queries: {queries_used}")
```

Code: txt

```txt
[*] Phase 2: Exploitation (budget: 400 queries)
  [P2 mid 200/400] top3=say:0.181, yeah:0.106, later:0.096
[+] Exploitation complete. Total queries: 800
```

Limiting to 20 test messages and the top 15 words concentrates testing on the most promising configurations. This focused approach ensures we get reliable estimates for our best words rather than spreading queries thinly across many mediocre options. Each test updates the word's score using exponential moving average, gradually refining effectiveness estimates. If "thanks" initially scored 0.15 but shows 0.18 in several exploitation tests, its score gradually increases to reflect this better performance.

This phase typically consumes exactly 400 queries, bringing our total to 800.

## Phase 3: Combination Discovery

The aim in combination search is to find small groups of words whose joint effect on spam probability is greater than testing single words alone. We search for pairs and triplets using the most promising words from Phase 2 and respect the remaining combination budget.

### Budget And Entry Conditions

We initialise remaining budget, create a store for the best combinations observed so far, and keep a simple counter of how many combinations we evaluated.

Code: python

```python
    # Phase 3: Combination discovery (allocated budget)
    remaining_combo = combination_budget
    print(f"\n[*] Phase 3: Combination search (budget: {remaining_combo} queries)")

    best_combinations = {}
    combos_tested = 0
```

### Putting The Search Together

We evaluate combinations on up to three messages to avoid overfitting to a single pattern. For each message, we ask `discover_word_combinations` to score single words, pairs, and triplets built from the Phase 2 shortlist. We then update our global store with the strongest observed score for each combination. Finally, we account for query usage approximately as two queries per tested combination and report a midpoint snapshot.

Code: python

```python
    if remaining_combo > 50:  # Need minimum queries for combinations
        for i in range(min(3, len(spam_messages))):
            if queries_used >= budget or remaining_combo <= 0:
                break

            test_msg = spam_messages[i]
            combos = discover_word_combinations(test_msg, top_word_list[:20], max_size=3)

            # Track best combinations across messages
            for combo, score in combos.items():
                if combo not in best_combinations or score > best_combinations[combo]:
                    best_combinations[combo] = score

            # Account for queries (~2 per combination) while respecting the budget
            to_add = min(remaining_combo, len(combos) * 2)
            queries_used += to_add
            remaining_combo -= to_add
            combos_tested += len(combos)

            # Midpoint snapshot
            if combination_budget > 0 and remaining_combo <= combination_budget // 2 and best_combinations:
                best = max(best_combinations.items(), key=lambda x: x[1])
                print(
                    f"  [P3 mid ~{combination_budget - remaining_combo}/{combination_budget}] "
                    f"combos_tested={combos_tested} | best={' + '.join(best[0])}:{best[1]:.3f}"
                )

            if remaining_combo <= 0:
                break
```

Code: txt

```txt
[*] Phase 3: Combination search (budget: 200 queries)
  [P3 mid ~130/200] combos_tested=65 | best=happy:0.000
  [P3 mid ~200/200] combos_tested=130 | best=later + going:0.000
[+] Combination search complete. Total queries: 1000
```

The `discover_word_combinations` helper returns effectiveness scores for single words, pairs, and triplets tested against the given message. We keep the strongest observed score per combination across messages. Under a fixed budget, this approach prioritises combinations that work on more than one message, improving reliability without exceeding cost.

### Final Results Compilation

Let's package the discoveries for use in attacks:

Code: python

```python
    print(f"[+] Combination search complete. Total queries: {queries_used}")

    # Return final results
    final_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)
    return final_words, best_combinations, queries_used
```

The function returns three values that enable flexible attack strategies. The final word rankings reflect all testing and updates throughout the three phases. The discovered combinations include synergistic groups that work better together than separately. The total queries consumed helps verify we stayed within budget constraints.

