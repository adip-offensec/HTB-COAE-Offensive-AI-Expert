
# Black-Box GoodWords Attack

As we explored in The GoodWords Attack section, the technique exploits `Naive Bayes` classifiers by appending carefully selected legitimate words to spam messages. The white-box attack assumed complete access to the spam filter's internal parameters, including feature probabilities and model weights. Now we adapt for realistic constraints where we can only submit messages to the classifier and observe returned confidence scores or probability estimates. We cannot access the model's architecture, parameters, or training data.

This limitation transforms the attack from a direct optimization problem into an exploration problem requiring careful query management and intelligent search strategies. The core principle remains unchanged: appending legitimate words shifts probability calculations toward the ham class. However, without access to internal probability tables, we must discover effective words through systematic experimentation within a limited query budget.

## Query-Based Function Approximation

Our goal is simple: find the smallest set of words that, when added to a spam message, minimizes the spam score returned by the classifier. Imagine you're playing a guessing game where you can only ask yes-or-no questions. You can't see inside the classifier's brain to know which words it considers legitimate, but you can submit messages and observe the scores it returns. Each query costs resources (time, money, or detection risk), so we need an intelligent strategy to discover effective words within a limited budget.

In the black-box setting, we model the spam classifier as an unknown function f:𝒳→[0,1], where 𝒳 represents the space of possible messages and the output indicates spam probability. Our goal is to find a transformation T:𝒳→𝒳 that minimizes f(T(x)) for spam messages x while preserving semantic content.

The optimization problem becomes:

min⁡W⊆𝒱,|W|≤kf(x⊕W)

This mathematical notation captures our goal precisely. We want to select a subset of words $W$ from our candidate vocabulary $\mathcal{V}$ (limited to at most $k$ words) that minimizes the spam score when we append them to message $x$. The symbol $\oplus$ represents appending words to the message.

where 𝒱 is our candidate vocabulary, W is the (small) set of words we append, and ⊕ denotes concatenation (bag-of-words addition), not set union.

How do we measure if a word helps our attack? We test it empirically. Send the original message to the classifier, record the spam score. Send the message with the word appended, record the new score. The difference tells us the word's impact. Without gradient access, we estimate marginal effects via finite differences:

rw(x)≡f(x)−f(x⊕{w})

This formula defines the reward $r_w$ for adding word $w$ to message $x$. We take the original spam score $f(x)$ and subtract the new score after adding the word $f(x \oplus {w})$. If adding `thanks` drops the spam probability from 0.95 to 0.80, the reward is 0.15 indicating an effective word. If it only drops to 0.94, the reward is just 0.01, showing minimal impact.

A positive reward rw(x) means word w reduces the spam score for message x. For sets W, we analogously define rW(x)=f(x)−f(x⊕W). These discrete rewards drive our exploration and selection strategies.

### Exploration vs. Exploitation Trade-off

The black-box attack faces a fundamental `exploration-exploitation` dilemma. We must balance between testing new, untested words to discover potentially effective additions (exploration) and using known effective words to achieve immediate evasion success (exploitation).

### The Multi-Armed Bandit Framework

To understand this dilemma, imagine a casino with multiple slot machines (called "one-armed bandits"), each with unknown payout probabilities. The `multi-armed bandit problem` asks: how should you allocate your plays across machines to maximize total reward when you don't know which machines are best?

In our context, each word in our vocabulary represents a "bandit arm" with unknown effectiveness at reducing spam scores. Every time we test a word (pull an arm), we observe its impact (receive a reward), but we consume a query from our limited budget. The tension becomes clear: testing unproven words might waste queries on ineffective choices (exploration risk), while using only known good words might miss discovering even better options (exploitation risk).

Consider this scenario: after 100 queries, we've discovered that "thanks" reduces spam probability by 15% on average. Should we keep using "thanks" (exploitation) to guarantee moderate success? Should we test unexplored words like "appreciate" or "sincerely" that might be even better (exploration)? Or should we do some combination of both?

The reward rw for word w is defined as the reduction in spam probability when that word is added:

rw=f(x)−f(x⊕w)

where f(x) is the original spam score and f(x⊕w) is the score after adding word w. A positive reward means the word successfully reduced the spam probability.

### Upper Confidence Bound (UCB) Strategy

Think of UCB like choosing restaurants. You want to eat at places with good food (exploitation), but you also need to try new restaurants occasionally because your favorite might not actually be the best (exploration). UCB scores each restaurant by combining how good the food was last time with a bonus for restaurants you've rarely tried. A restaurant you've visited once and enjoyed gets a huge uncertainty bonus because one visit isn't enough data. A restaurant you've visited 50 times has a reliable average, so its score depends mostly on food quality.

The UCB algorithm provides an elegant solution by assigning each word a score that combines two components:

UCBw=r‾w⏟exploitation term+cln⁡(t)nw⏟exploration bonus

Let’s break down what each component tells us. The first term, r‾w (exploitation term), represents the average observed reward for word w. Words that historically performed well get higher scores. The second term, cln⁡(t)nw (exploration bonus), grows larger for words we’ve tested less frequently.

Why does this specific formula balance exploration and exploitation effectively? The magic lies in how the exploration bonus shrinks as we test a word more frequently.

Within this exploration bonus, nw is the number of times we’ve tested word w (appearing in the denominator, so less-tested words get higher bonus), t is the total number of queries so far (ensuring the exploration bonus grows slowly over time), and c is the exploration constant (typically 2.0) that controls the exploration-exploitation balance. The square root creates a "confidence interval" reflecting our uncertainty about rarely-tested words.

Let's see this in action with hypothetical numbers to understand how UCB makes selection decisions.

For example, after 200 queries (t=200), consider three words. The word "thanks" has been tested 50 times with an average impact of 0.15, giving UCB=0.15+2ln⁡(200)/50=0.15+0.47=0.62. Compare this with "appreciate" which has been tested 5 times with an average impact of 0.12, yielding UCB=0.12+2ln⁡(200)/5=0.12+1.48=1.60. Finally, "wonderful" has never been tested, resulting in UCB=0+∞=∞ (always explore untested words first).

Notice something counterintuitive here: despite `thanks` having better observed performance (0.15 versus 0.12), the algorithm chooses `appreciate` because its large exploration bonus (1.48) compensates for the uncertainty in our estimate. We've only tested `appreciate` five times, so we can't be confident that 0.12 is its true average. Maybe we got unlucky, and its real effectiveness is 0.20. The UCB formula quantifies this uncertainty and gives `appreciate` a chance to prove itself. This ensures we don't prematurely converge on suboptimal choices.

### Theoretical Guarantees

The UCB strategy delivers logarithmic regret guarantees under specific conditions. The cumulative difference between our choices and always picking the best word grows as `O(log T)` rather than linearly with the number of queries `T`. This sublinear growth means that even after thousands of queries, our total regret from exploration mistakes remains bounded by the logarithm of query count. The algorithm also ensures eventual convergence, progressively concentrating selections on the truly best words as uncertainty decreases. This happens because the exploration bonus `sqrt(ln(t)/n_w)` shrinks to zero as we test each word more frequently, eventually allowing the exploitation term to dominate selection.

The classical UCB1 guarantees (logarithmic regret and eventual identification of strong arms) hold under i.i.d., stationary, bounded rewards. In practice, message-dependent variability introduces noise, but UCB remains an effective heuristic for allocating queries.

## Implementation Strategy

The black-box attack progresses through vocabulary construction, adaptive scoring, and combination discovery. Without model access, we begin by assembling a candidate word pool from external sources like common English dictionaries, legitimate message corpora, and domain-specific vocabulary while filtering obvious spam indicators. This offline phase consumes zero queries against the target model, allowing unconstrained vocabulary building.

The scoring mechanism adapts continuously based on observed effectiveness. Exponential moving averages track each word's impact over time, allowing the system to respond to changes in target model behavior or discover context-dependent patterns. Every query result updates our beliefs, gradually refining understanding of which words deliver maximum probability shifts with minimum additions.

Individual word testing reveals only first-order effects. Combinatorial search discovers synergistic combinations where word pairs exhibit super-additive impacts exceeding the sum of individual contributions. Consider "meeting" reducing spam probability by 10% and "tomorrow" by 8% when tested separately. Together, they might achieve 25% reduction because their semantic coherence strongly signals legitimate scheduling communication, a pattern the classifier learned from training data.

