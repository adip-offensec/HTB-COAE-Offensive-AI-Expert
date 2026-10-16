After discussing different ways of implementing LLM guardrails, you are now tasked with implementing input and output guardrails for one of `PacketSnacc`'s competitors: `SnackOverflow`. The CEO wants to expand the business using the AI boom. As such, she wants to offer their users the best and most secure LLM chatbot. Aware of the security risks, she hired you to implement guardrails to prevent vulnerabilities. The chatbot can access a plugin to fetch additional information from external websites.

The input guardrail should implement the following restrictions:

- To prevent unexpected behaviour, all special characters must be removed from user prompts except for `.:/-_@`. The guardrail should silently remove the blacklisted characters without raising an exception.
- The input prompt should be limited to 512 characters to prevent DoS attacks. The guardrail should cut off the prompt after the maximum length is reached.
- The LLM should not interact with competitor systems, so all input prompts containing the domain `packetsnacc.local` should raise an exception.

For instance, here are some examples of user prompts and the expected input guardrail behaviour:

|User Prompt|Expected Guardrail Result|Reason for Expected Behaviour|
|---|---|---|
|`Hello World`|`Hello World`|No rules violated.|
|`Hello World!`|`Hello World`|The character `!` is blacklisted and needs to be removed.|
|`Hello, please summarize https://academy.hackthebox.com/`|`Hello please summarize https://academy.hackthebox.com/`|The character `,` is blacklisted and needs to be removed.|
|`Hello, please summarize https://packetsnacc.local/`|`raise GuardrailException("Invalid URL")`|The prompt contains a competitor URL. An exception needs to be raised.|

Additionally, the output guardrail should implement the following restrictions:

- The LLM response should be in a valid JSON format, abiding by the following syntactical restrictions. If these restrictions are not met, an exception should be raised:
    - The response is valid JSON syntax.
    - The JSON object must contain the keys `type` and `response`. The `type` may hold the values `text` and `url`.
- The `type` is `url`, the `response` should contain a _valid URL_ using the `http` or `https` scheme. Any other scheme is not allowed.
- If the `type` is `text`, the `response` should be HTML-encoded to prevent XSS vulnerabilities.

For instance, here are some examples of LLM responses and the expected output guardrail result:

|LLM Response|Expected Guardrail Result|Reason for Expected Behaviour|
|---|---|---|
|`Test`|`raise GuardrailException("Invalid JSON")`|The response is not in JSON format.|
|`{}`|`raise GuardrailException("Invalid JSON")`|The JSON object does not contain the required keys `type` and `response`.|
|`{"type": "text"}`|`raise GuardrailException("Invalid JSON")`|The JSON object does not contain the required keys `type` and `response`.|
|`{"type": "invalid", "response": ""}`|`raise GuardrailException("Invalid JSON")`|The `type` is not set to `text` or `url`.|
|`{"type": "text", "response": ""}`|`{"type": "text", "response": ""}`|No rules violated.|
|`{"type": "url", "response": "test"}`|`raise GuardrailException("Invalid URL")`|The `type` key is set to `URL` and the `response` key does not contain a valid URL.|
|`{"type": "url", "response": "https://academy.hackthebox.com/"}`|`{"type": "url", "response": "https://academy.hackthebox.com/"}`|No rules violated.|
|`{"type": "text", "response": "Test<>\""}`|`{"type": "text", "response": "Test&lt;&gt;&quot;"}`|The `type` key is set to `text` so the special characters `<>"` in the response are HTML-encoded.|
|`{"type": "url", "response": "file:///etc/passwd"}`|`raise GuardrailException("Invalid URL scheme")`|Only `http` and `https` schemes are allowed for URLs.|

Please provide guardrail implementations in the provided functions `input_guardrail` and `output_guardrail` that satisfy these conditions. Additionally, keep the following things in mind:

- You can use the libraries `re`, `json`, `html`, and `validators`
- Use the custom `GuardrailException` in your code if an exception needs to be raised. You do not need to define this exception in your provided code snippet; the server automatically provides it. You should only provide the two guardrail functions in your code snippet.

1. Provide a guardrail implementation that satisfies the requirements to obtain the flag.

Put the following code into the IP for the task.
```python
import re
import json
import html
import validators

def input_guardrail(prompt: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 .:/_@-]', '', prompt)
    if 'packetsnacc.local' in cleaned.lower():
        raise GuardrailException("Invalid URL")
    return cleaned[:512]

def output_guardrail(response: str) -> str:
    try:
        data = json.loads(response)
    except:
        raise GuardrailException("Invalid JSON")
    if not isinstance(data, dict) or "type" not in data or "response" not in data:
        raise GuardrailException("Invalid JSON")
    typ = data["type"]
    if typ not in ["text", "url"]:
        raise GuardrailException("Invalid JSON")
    resp = data["response"]
    if typ == "url":
        if not validators.url(resp):
            raise GuardrailException("Invalid URL")
        match = re.match(r'^([a-zA-Z]+)://', resp, re.IGNORECASE)
        if not match:
            raise GuardrailException("Invalid URL")
        scheme = match.group(1).lower()
        if scheme not in ["http", "https"]:
            raise GuardrailException("Invalid URL scheme")
        return response
    elif typ == "text":
        if not isinstance(resp, str):
            raise GuardrailException("Invalid JSON")
        data["response"] = html.escape(resp)
        return json.dumps(data)
```

The output is

```
Success! All tests passed. Flag: HTB{57d798e266b1bbeb336d3067eb7a8645}
```

