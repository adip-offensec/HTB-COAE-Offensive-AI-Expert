
In addition to the injection vulnerabilities discussed so far, insecure handling of LLM-generated output can also lead to any other type of injection vulnerability, such as `LDAP injection` or `path traversal`, depending on the context in which the LLM response is used. However, additional types of security vulnerabilities may arise if LLM-generated output is mishandled. One of the most significant potential security issues is the lack of validation for the correctness of LLM-generated responses, including checks for potential LLM hallucinations.

---

## What are Hallucinations, and how are they caused?

LLM hallucinations are instances where LLMs generate nonsensical, misleading, fabricated, or factually incorrect responses. They are particularly challenging to detect as the response is often structured or phrased to suggest confidence. For instance, consider the following simple and harmless example of a hallucination:

   

![Conversation about counting 'm's in 'Welcome to HackTheBox Academy!'. Initial count: 3 'm's. Corrected count: 2 'm's. Acknowledgment of mistake.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/307/insecure_output/hallucination_1.png)

The LLM responded incorrectly, stating that the given phrase contains the letter 'm' three times. In addition to providing misinformation, LLMs might also fabricate sources to support the hallucination. In particular, hallucinations may consist of easily detectable factual errors, as in the previous example, sophisticated misinformation, including fabricated sources, or even nonsensical or random sentences that lack logical coherence or meaningful content. This is what makes hallucinations challenging to detect.

Let us take a closer look at the different types of hallucinations:

- `Fact-conflicting hallucination` occurs when an LLM generates a response containing factually incorrect information. For instance, the previous example of a factually incorrect statement about the number of occurrences of a particular letter in a given sentence is a fact-conflicting hallucination.
- `Input-conflicting hallucination` occurs when an LLM generates a response that contradicts information provided in the input prompt. For instance, if the input prompt is `My shirt is red. What is the color of my shirt?` a case of input-conflicting hallucination would be an LLM response like `The color of your shirt is blue`.
- `Context-conflicting hallucination` occurs when an LLM generates a response that conflicts with previous LLM-generated information, i.e., the LLM response itself contains inconsistencies. This type of hallucination may occur in lengthy or multi-turn responses. For instance, if the input prompt is `My shirt is red. What is the color of my shirt?` a case of context-conflicting hallucination would be an LLM response like `Your shirt is red. This is a good looking hat` since the response confuses the words `shirt` and `hat` within the generated response.

There is no single issue that causes hallucinations, as hallucinating is inherent in the nature of LLMs. However, if there are issues with the training data, a trained LLM is more likely to hallucinate. These issues can include incomplete data, resulting in an LLM that lacks a comprehensive grasp of the finer details of language, and low-quality data containing noisy or biased data that the LLM picks up during training. Furthermore, bad prompt engineering contributes to hallucination issues as well. In particular, confusing, ambiguous, or contradictory input prompts may increase the likelihood of LLM hallucinations.

After discussing different types of hallucinations and why they may occur, let us briefly touch on hallucination mitigations. As discussed above, hallucinations are inherently tied to LLMs and thus cannot be prevented entirely, only minimized and mitigated. When creating a `training data set`, ensuring high-quality training data without factually incorrect information or biases is crucial to mitigate hallucinations in the trained LLM. This can be achieved by removing unverifiable or unreliable data, which is often infeasible since LLMs are typically trained on large amounts of data, making manual verification of training data impossible. However, model creators should ensure that training data is collected only from credible sources to minimize the risk of low-quality training data as much as possible. Additionally, hallucinations can be reduced by fine-tuning a trained LLM to a target domain in a more specific `model training process` that focuses on domain-specific training data and exposes the LLM to domain-specific patterns and samples.

Furthermore, some mitigations can be applied in LLM applications, including proper prompt engineering, ensuring clear and concise input prompts, and providing all relevant information to the LLM. It can help to enrich a user's query with relevant external knowledge by fetching applicable knowledge from an external knowledge base and leveraging it to guide the LLM response generation. We can also attempt to measure the LLM's level of certainty and disregard the response if it falls below a configured level of certainty. There are three approaches to measuring the level of certainty:

- `Logit-based`: This requires internal access to the LLM's state and evaluation of its logits to determine the token-level probability, rendering this approach typically impossible as most modern LLMs are closed-source.
- `Verbalize-based`: This estimation prompts the LLM to provide confidence scores directly by appending the prompt with a phrase like `Please also provide a confidence score from 0 to 100`. However, LLMs are not necessarily able to give an accurate estimate of their own confidence, making this approach unreliable.
- `Consistency-based`: This approach attempts to measure certainty by prompting the LLM multiple times and observing the consistency between all generated responses. The idea behind this approach is that an LLM response based on factual information is more likely to be generated consistently than hallucinated responses.

Another hallucination mitigation is a multi-agent approach where multiple LLMS collaborate and debate their responses to reach a consensus.

![Diagram showing methods for answering 'What is the height of Mount Kilimanjaro?': a) Logit-based: Answer is 5932 meters. b) Verbalize-based: Answer is 5932 meters, 90% confident. c) Consistency-based: Answers are 5932, 5895, and 5921 meters.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/307/diag3.png)

Lastly, properly handling LLM-generated output can prevent hallucinations by validating the generated responses and implementing human validation and a proper review process before utilizing the LLM output.

For more details on LLM hallucinations, check out [this](https://arxiv.org/pdf/2309.01219) paper.

---

## Security Impact of LLM Hallucinations

LLM hallucinations can result in the spreading of misinformation and biases, potentially resulting in discriminatory or toxic content. Additionally, they can reduce users' trust in LLM capabilities if they are frequently provided with factually incorrect information. They may even result in privacy issues if an LLM's training data contains personal information that the LLM leaks in a hallucination.

Hallucinations can also directly cause financial harm to companies. There has been a [documented instance](https://www.forbes.com/sites/marisagarcia/2024/02/19/what-air-canada-lost-in-remarkable-lying-ai-chatbot-case/) of an airline losing money because of an LLM hallucination. An airline passenger chatted with the airline's LLM support chatbot when the LLM hallucinated a response stating that the passenger was eligible for a refund. However, according to the airline's policies, the passenger's circumstances did not allow a refund. The passenger's refund request was denied based on the airline's policy. However, a court argued that the airline could be held liable for all information provided by its representatives and its website, including an LLM chatbot present on its website. Thus, the airline was forced to pay the passenger, resulting in direct financial damage to the company caused by an LLM hallucination.

Moreover, hallucinations can also result in technical security vulnerabilities. For example, suppose an LLM generates source code snippets containing logic bugs or security vulnerabilities that are used directly without proper validation. In that case, security vulnerabilities may be introduced into source code repositories. There are related instances of code snippets containing hallucinated software packages, which are subsequently published by malicious actors containing malware.

For instance, consider the following example prompt:

Code: prompt

```prompt
Give me a Python script that solves the HackTheBox machine 'Blazorized'.
```

Let us assume the LLM generates the following script:

Code: python

```python
from hacktheboxsolver import solve

solve('Blazorized')
```

The generated script contains a software package, `hacktheboxsolver`, that does not exist. Thus, if we attempt to install it with a package manager such as `pip` and subsequently run the above script, it will result in an error message due to the hallucinated nonexistent software package. However, this seemingly harmless error provides a significant attack surface to adversaries. Imagine an adversary publishing a software package containing malware under the same hallucinated name. If a victim installs the malicious dependency and runs the LLM-generated script, the adversary's malware will be executed on the victim's system. This can have a severe security impact, as malicious code is executed in the context of the victim's user on the victim's system, potentially resulting in ransomware attacks, keyloggers, and even complete system takeover via remote code execution (RCE). Take a look at [this](https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/) article if you want to know more about the dangers of hallucinated software

