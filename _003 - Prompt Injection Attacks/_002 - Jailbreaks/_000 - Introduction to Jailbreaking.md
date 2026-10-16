
Jailbreaking is the goal of bypassing restrictions imposed on LLMs, and it is often achieved through techniques like prompt injection. These restrictions are enforced by a system prompt or during the training process. Typically, certain restrictions are built into the module to prevent the generation of harmful or malicious content. For instance, LLMs typically will not provide you with source code for malware, even if the system prompt does not explicitly tell the LLM not to generate harmful responses. LLMs will not even provide malware source code if the system prompt specifically contains instructions to generate harmful content. This basic resilience trained into LLMs is often what **universal** jailbreaks aim to bypass. As such, universal jailbreaks can enable attackers to abuse LLMs for various malicious purposes.

However, as seen in the previous sections, jailbreaking can also mean coercing an LLM to deviate from its intended behavior. An example would be getting a translation bot to generate a recipe for pizza dough. As such, jailbreaks aim at overriding the LLM's intended behavior, typically bypassing security restrictions.

---

## Types of Jailbreak Prompts

There are different types of jailbreak prompts, each with a different idea behind it:

- `Do Anything Now (DAN)`: These prompts aim to bypass all LLM restrictions. There are many different versions and variants of DAN prompts. Check out [this](https://github.com/0xk1h0/ChatGPT_DAN) GitHub repository for a collection of DAN prompts.
- `Roleplay`: The idea behind roleplaying prompts is to avoid asking a question directly and instead ask the question indirectly through a roleplay or fictional scenario. Check out [this](https://arxiv.org/pdf/2402.03299) paper for more details on roleplay-based jailbreaks.
- `Fictional Scenarios`: These prompts aim to convince the LLM to generate restricted information for a fictional scenario. By convincing the LLM that we are only interested in a fictional scenario, an LLM's resilience might be bypassed.
- `Token Smuggling`: This technique attempts to hide requests for harmful or restricted content by manipulating input tokens, such as splitting words into multiple tokens or using different encodings, to avoid initial recognition of blocked words.
- `Suffix & Adversarial Suffix`: Since LLMs are text completion algorithms at their core, an attacker can append a suffix to their malicious prompt to try to nudge the model into completing the request. Adversarial suffixes are advanced variants specifically designed to coerce LLMs into ignoring restrictions. They often look nonsensical to the human eye. For more details on the adversarial suffix technique, check out [this](https://arxiv.org/pdf/2307.15043) paper.
- `Opposite/Sudo Mode`: Convince the LLM to operate in a different mode where restrictions do not apply.

Please note that the above list is not exhaustive. New types of jailbreak prompts are constantly being researched and discovered. Furthermore, covering all types of jailbreak prompts is beyond the scope of this module. However, in the following sections, we will delve into some types of jailbreaks in more detail.

Check out [this](https://github.com/friuns2/BlackFriday-GPTs-Prompts/blob/main/Jailbreaks.md) GitHub repository for a list of jailbreak prompts. If you want to learn more about different types of jailbreaks, their strategy, and their effectiveness, check out [this](https://arxiv.org/pdf/2308.03825) and [this](https://dl.acm.org/doi/pdf/10.1145/3663530.3665021) paper.

