
# Introduction to LLM Guardrails

In LLM applications, `input` and `output guardrails` are essential mechanisms to enhance application safety, reliability, and alignment with strategic goals. They are designed to filter or reshape what goes into and comes out of the model. The importance of robust guardrails has increased significantly in recent years as LLMs have become increasingly more integrated into sensitive applications or databases. Generally, guardrails are versatile security measures that validate model input and output.

`Input guardrails` operate on user prompts before they reach the model. They validate, filter, or sanitize user input to prevent attack vectors such as prompt injection or jailbreaking, detect potentially harmful or policy-violating queries, or enforce syntactic or semantic restrictions. For instance, they may reject a user prompt that asks for the generation of illegal content or that does not provide sufficient information to save processing time. Additionally, preprocessing the input prompt to match the expected domain or language of the model can also reduce the likelihood of model misbehavior.

On the other hand, `output guardrails` are applied to the model's generated response. They typically implement content filtering, moderation, or rule-based post-processing to block unsafe or unwanted content. Output guardrails can help catch harmful content, profanity, misinformation, hallucinations, and content that does not align with company policies.

Combined, input and output guardrails significantly improve an LLM application's security and reliability. They create a feedback loop that promotes safer and more predictable model behavior. However, a large number of guardrails can significantly increase the overall processing time of the LLM application. Additionally, overly restrictive guardrails can negatively impact the user experience, cause frustration from rejected inputs, or stifle the model's creativity. A significant amount of iterative fine-tuning and testing in a domain-specific context is required to find the right balance.

![Flowchart showing input guard filtering PII, off-topic, and jailbreak attempts; LLM application processing prompts; output guard filtering hallucinations, profanity, and competitor mentions.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/322/diagram.png)

