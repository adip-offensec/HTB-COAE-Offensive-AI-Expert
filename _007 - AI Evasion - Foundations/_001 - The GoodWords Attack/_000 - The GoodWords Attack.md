
# The GoodWords Attack

The GoodWords attack, first introduced by Lowd and Meek in their 2005 paper "[Good Word Attacks on Statistical Spam Filters](https://www.ceas.cc/papers-2005/125.pdf)," is an adversarial technique designed to exploit fundamental probabilistic assumptions within `Naive Bayes` classifiers, particularly those deployed for spam detection. This attack methodology manipulates the classifier's decision-making process by strategically appending carefully selected legitimate words to malicious messages, disguising spam content with legitimate-looking terms.

At its core, the attack leverages the mathematical foundation of `Naive Bayes` classification, specifically targeting the classifier's reliance on word frequency distributions and conditional probabilities. The technique demonstrates how probabilistic models can be deceived through simple manipulations that exploit their inherent assumptions about feature independence and probability calculations.

The GoodWords attack is simple and effective. Rather than attempting to obfuscate or modify the malicious content itself, the attack preserves the original spam message intact while augmenting it with additional tokens that shift the overall probability distribution. This approach maintains the semantic meaning and intent of the spam message while simultaneously convincing the classifier that the message belongs to the legitimate class.

## Theoretical Foundation: Naive Bayes Classification Refresher

As we explored in the `Fundamentals of AI` module and saw applied in the `Applications of AI in InfoSec` module, `Naive Bayes` classifiers use `Bayes' theorem` to calculate the probability of a message belonging to a particular class. Here we'll briefly revisit the key concepts essential for understanding how the GoodWords attack exploits this algorithm.

Remember that `Naive Bayes` applies `Bayes' theorem` to classification:

P(C|D)=P(D|C)⋅P(C)P(D)

In spam detection, this calculates the probability that a message belongs to class C (`spam` or `ham`) given document D. As we covered earlier, the classifier’s decision rule compares posterior probabilities:

class=arg⁡max⁡c∈{spam,ham}P(c|D)

The critical "naive" assumption that makes this algorithm both efficient and vulnerable is the conditional independence of features. This allows the likelihood to factorize as:

P(D|C)=∏i=1nP(wi|C)

where each word wi contributes independently to the probability calculation. To avoid numerical underflow, implementations work in log space:

class=arg⁡max⁡c∈{spam,ham}[logP(c)+∑i=1nlogP(wi|c)]

This independence assumption, while making `Naive Bayes` computationally efficient, creates a vulnerability. The additive nature of log probabilities means that classification decisions can be influenced by simply adding more words to a message. With this understanding of the underlying probabilistic framework, we can now examine the specific mathematical mechanics that make the GoodWords attack so effective.

## Attack Mechanics and Probability Manipulation

Think of the classifier as a scale weighing evidence for spam versus ham. Each word adds weight to one side or the other. Spam words like `FREE` and `WINNER` pile evidence onto the spam side. The attack works by adding enough legitimate words to the ham side that the scale tips, even though the original spam content remains unchanged. The beauty lies in how these weights combine through simple addition in log space, making the attack both predictable and powerful.

The attack’s effectiveness comes from its manipulation of the probability calculations within the `Naive Bayes` algorithm. When we consider a spam message Mspam with words {w1,w2,...,wm}, the classifier normally calculates:

log⁡P(spam|Mspam)=log⁡P(spam)+∑i=1mlog⁡P(wi|spam)log⁡P(ham|Mspam)=log⁡P(ham)+∑i=1mlog⁡P(wi|ham)

These equations show how the classifier builds its decision. Start with the base probability that any message is spam or ham, then add up the evidence from each word. In plain terms, if a message contains `FREE`, `WINNER`, and `CLAIM`, each word contributes spam evidence that gets summed together. For successful classification as spam, we need:

log⁡P(spam|Mspam)>log⁡P(ham|Mspam)

Now here's where the attack comes in. We take that same spam message and append carefully chosen legitimate words.

The attack modifies the message by appending a set of "good words" G={g1,g2,...,gk}, creating an augmented message Maugmented=Mspam∪G. The new probability calculations become:

log⁡P(spam|Maugmented)=log⁡P(spam)+∑i=1mlog⁡P(wi|spam)+∑j=1klog⁡P(gj|spam)

log⁡P(ham|Maugmented)=log⁡P(ham)+∑i=1mlog⁡P(wi|ham)+∑j=1klog⁡P(gj|ham)

Notice the new terms at the end of each equation. The original spam words are still there, contributing the same evidence they always did. But now we've added good words that strongly favor ham. Each good word adds more weight to the ham side of the scale than to the spam side. Add enough of them, and the scale tips. The attack succeeds when the addition of good words reverses the inequality, causing:

log⁡P(ham|Maugmented)>log⁡P(spam|Maugmented)

When does this reversal happen? We need the good words to contribute more ham evidence than the original message contributed spam evidence. Mathematically, this reversal occurs when:

∑j=1k[log⁡P(gj|ham)−log⁡P(gj|spam)]>∑i=1m[log⁡P(wi|spam)−log⁡P(wi|ham)]+[log⁡P(spam)−log⁡P(ham)]

Breaking this down, the left side represents the net contribution of our good words. Each good word shifts the balance toward ham by the difference between its ham probability and spam probability. If `meeting` appears frequently in legitimate messages but rarely in spam, it contributes strongly to the left side. The right side represents the hurdle we need to overcome: the original spam signal plus any inherent bias the classifier has toward one class or another.

For example, imagine a message with strong spam words contributing a combined difference of 8.0 toward spam. If we add words like `meeting`, `tomorrow`, and `thanks` that each contribute 3.0 toward ham, we need at least three such words (3 × 3.0 = 9.0) to overcome the spam signal and tip the classification.

The attack’s goal is to select words gj that maximize the difference log⁡P(gj|ham)−log⁡P(gj|spam), effectively overwhelming the spam signal with legitimate-looking features.

## Good Word Selection Strategy

The effectiveness of the GoodWords attack depends on selecting appropriate words to append. The optimal selection strategy identifies words with strong discriminative power in favor of the legitimate class. We define a "goodness score" for each word w as:

S(w)=P(w|ham)P(w|spam)+ϵ

where ϵ is a small constant to prevent division by zero. Words with high goodness scores appear frequently in legitimate messages while being rare or absent in spam messages. Consider an example: if the word "meeting" appears in 45

![The Goodness Prism: A triangle with arrows showing words 'thanks,' 'the,' and 'free' scoring high, low, and bad.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/good_word_selection_strategy_panel2_prism.png)

We employ a systematic approach to word selection in our attack implementation. First, we analyze the training corpus to compute conditional probabilities for all vocabulary terms. We then rank words by their goodness scores, creating a prioritized list of candidates. This ranking considers not only the raw probability ratios but also the absolute frequencies to ensure selected words are representative of legitimate communication patterns.

![The Two Vocabulary Worlds: Venn diagram with 'Legitimate (Ham)' words like 'thanks' and 'meeting,' 'Spam' words like 'free' and 'prize,' and shared words 'the' and 'is' with low discriminative power.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/318/good_word_selection_strategy_panel1_worlds.png)

When selecting words, we must avoid those that might trigger other spam detection heuristics. While a word might have a high goodness score, its inclusion could activate rule-based filters or appear suspicious to human reviewers. The selection process often incorporates additional constraints to ensure the augmented message maintains linguistic coherence and plausibility.

The number of words to add represents an important parameter balancing attack effectiveness against message length constraints. Adding too few words may fail to shift the probability sufficiently, while adding too many could make the message appear unnatural or exceed transmission limits. Analysis shows that most spam messages can be successfully misclassified by adding between 15 and 30 carefully selected good words, achieving evasion rates over 90%.

## Why the Attack Works: Exploiting Independence Assumptions

The defining assumption of `Naive Bayes` - conditional independence of words given the class - is both its strength and weakness. This simplification is computationally efficient and often effective in practice, yet it also explains why the `GoodWords` attack works. Because the likelihood factorizes as:

P(D∣C)=∏i=1nP(wi∣C)

Each word contributes independently to the total score. In `log space` the contributions add rather than multiply, so additional words simply shift the running sum. An attacker can therefore append words that have strongly skewed conditional probabilities toward the legitimate class, incrementally pushing the decision toward `ham` even when the original content is characteristic of `spam`.

The classifier does not model word relationships or broader context, so it cannot penalize the semantic mismatch between the appended legitimate words and the underlying spammy message. Given enough favorable additions, quantity overwhelms quality, and the accumulated evidence overturns the initial spam signal.

Several implementation choices increase this susceptibility. Training typically fixes word-class probabilities from historical data, producing relatively static distributions that adversaries can study. Once those distributions are understood, the most advantageous `GoodWords` for augmentation can be identified systematically. Standard smoothing methods such as `Laplace smoothing` assign non-zero probability to unseen or rare words, which prevents rejection of unusual combinations and increases the range of words available to an attacker. Together, the independence assumption, the additive scoring in `log space`, the static learned distributions, and the effects of smoothing create the conditions that the `GoodWords` attack exploits.

