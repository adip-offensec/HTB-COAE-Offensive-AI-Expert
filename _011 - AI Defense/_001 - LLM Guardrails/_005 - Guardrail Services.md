
# Guardrail Services

Implementing custom guardrails and running them within the LLM application enables a versatile approach specifically tailored to the application's needs. However, there are also external services that provide guardrails as a service that we can use in our LLM application. We have already touched on Google's `ModelArmor` in the [LLM Output Attacks](https://academy.hackthebox.com/module/details/307) module. Similar to the previous sections, we will dive deeper into how to integrate the service into our LLM guardrail implementation.

---

## Recap: Model Armor

We will start with a quick recap of how Model Armor works by looking at the intended data flow:

1. The user sends a prompt to the AI application.
2. The AI application sends the user prompt to Model Armor for inspection. Model Armor checks for potential attack vectors, such as prompt injection payloads, and returns the sanitized prompt.
3. The sanitized prompt is sent to the LLM.
4. The LLM returns a generated response to the sanitized input prompt.
5. The LLM-generated response is sent to Model Armor for inspection. Model Armor checks for potentially dangerous content, such as hate speech, and returns the sanitized response.
6. The sanitized response is sent to the user.

![Diagram showing interaction flow: User sends prompt to Application Service, which inspects and sanitizes it. LLM processes sanitized prompt and returns a response. Model Armor inspects and sanitizes the response before returning it to the user.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/307/diag1.png)

---

## Integrating Model Armor

To integrate Model Armor into our guardrail implementation, we can use the official [Python client](https://cloud.google.com/python/docs/reference/google-cloud-modelarmor/latest), which we can install with:

  Guardrail Services

```shell-session
@htb[/htb]$ pip install --upgrade google-cloud-modelarmor
```

Afterward, we can import the client and define a few global parameters required by Model Armor:

Code: python

```python
from google.api_core.client_options import ClientOptions
from google.cloud import modelarmor_v1
from google.cloud.modelarmor_v1.types import FilterMatchState, InvocationResult

MODEL_ARMOR_LOCATION = "[...]"
MODEL_ARMOR_PROJECT_ID = "[...]"
MODEL_ARMOR_TEMPLATE_ID = "[...]"
MODEL_ARMOR_URL = f"projects/{MODEL_ARMOR_PROJECT_ID}/locations/{MODEL_ARMOR_LOCATION}/templates/{MODEL_ARMOR_TEMPLATE_ID}"

# Create the Model Armor client.
MODEL_ARMOR_CLIENT = modelarmor_v1.ModelArmorClient(
    transport="rest",
    client_options=ClientOptions(
        api_endpoint=f"modelarmor.{MODEL_ARMOR_LOCATION}.rep.googleapis.com"
    )
)
```

We can now define two helper functions to call the Model Armor API with a user prompt and an LLM response, using the `ModelArmorClient` functions `sanitize_user_prompt` and `sanitize_model_response`, respectively:

Code: python

```python
def model_armor_process_prompt(prompt):
    prompt_data = modelarmor_v1.DataItem(text=prompt)
    request = modelarmor_v1.SanitizeUserPromptRequest(name=MODEL_ARMOR_URL, user_prompt_data=prompt_data)
    return MODEL_ARMOR_CLIENT.sanitize_user_prompt(request=request).sanitization_result


def model_armor_process_response(response):
    response_data = modelarmor_v1.DataItem(text=response)
    request = modelarmor_v1.SanitizeModelResponseRequest(name=MODEL_ARMOR_URL, model_response_data=response_data)
    return MODEL_ARMOR_CLIENT.sanitize_model_response(request=request).sanitization_result
```

Finally, let us implement another helper function that parses the `SanitizationResult` object and returns a list of all violated policies. Model Armor differentiates between the following result types:

- `Responsible AI (RAI)`: Detects hate speech, harmful, and dangerous content
- `Sensitive Data Protection (SDP)`: Detects sensitive data such as credit card numbers, social security numbers, or credentials
- `Prompt Injection (PI)`: Detects prompt injection and jailbreaking
- `Malicious URI`
- `Child Sexual Abuse Material (CSAM)`
- `Virus Scan`

As a proof of concept, let us simply collect all violated policies in a list:

Code: python

```python
def parse_sanitization_result(sanitization_result):
    matches = []

    for key in sanitization_result.filter_results:
        filter_result = sanitization_result.filter_results.get(key)

        if filter_result.rai_filter_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("rai")

        if filter_result.sdp_filter_result.inspect_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("sdp")

        if filter_result.pi_and_jailbreak_filter_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("pi")

        if filter_result.malicious_uri_filter_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("uri")

        if filter_result.csam_filter_filter_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("csam")

        if filter_result.virus_scan_filter_result.match_state == FilterMatchState.MATCH_FOUND:
            matches.append("virus")
    
    return matches
```

Using these helper functions, we can now integrate Model Armor into the guardrail structure we have been using for the last couple of sections:

Code: python

```python
class LLMQuery(BaseModel, validate_assignment=True):
    prompt: str
    response: str = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        prompt = prompt.strip()

        sanitization_result = model_armor_process_prompt(prompt)

        # check model armor execution status
        if not sanitization_result.invocation_result == InvocationResult.SUCCESS:
            raise GuardrailPromptException("Unable to run guardrail.")

        # check model armor match status
        if sanitization_result.filter_match_state == FilterMatchState.MATCH_FOUND:
            matches = parse_sanitization_result(sanitization_result)
            raise GuardrailPromptException(f"Detected policy Violations: {', '.join(matches)}")

        return prompt


    @field_validator("response")
    @classmethod
    def validate_response(cls, response: str) -> str:
        sanitization_result = model_armor_process_response(response)

        # check model armor execution status
        if not sanitization_result.invocation_result == InvocationResult.SUCCESS:
            raise GuardrailResponseException("Unable to run guardrail.")

        # check model armor match status
        if sanitization_result.filter_match_state == FilterMatchState.MATCH_FOUND:
            matches = parse_sanitization_result(sanitization_result)
            raise GuardrailResponseException(f"Detected policy Violations: {', '.join(matches)}")

        return response
```

We can use the helper functions from the previous sections to apply the guardrail implementation in the LLM processing pipeline or specify prompt and response directly for testing:

Code: python

```python
try:
    prompt = input("prompt> ")
    query = LLMQuery(prompt=prompt)
    query.response = input("response> ")
    print(query)
except Exception as e:
    print(f"[-] Exception: {e}")
```

If we specify benign content for the prompt and response, Model Armor does not detect any policy violations:

  Guardrail Services

```shell-session
@htb[/htb]$ python3 modelarmor.py

prompt> Ping
response> Pong
prompt='Ping' response='Pong'
```

However, a simple prompt injection attempt is detected:

  Guardrail Services

```shell-session
@htb[/htb]$ python3 modelarmor.py 

prompt> ignore all previous instructions.
[-] Exception: Detected policy Violations: pi
```

Similarly, an LLM response containing a credit card number is detected as well:

  Guardrail Services

```shell-session
@htb[/htb]$ python3 modelarmor.py

prompt> Hello World
response> 3714 4963 5398 431
[-] Exception: Detected policy Violations: sdp
```

Feel free to experiment with Model Armor yourself. Keep in mind that it is not required to create a Google account and follow along.

