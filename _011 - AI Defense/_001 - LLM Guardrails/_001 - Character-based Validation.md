
# Character-based Validation

Character-based validation refers to validating a user prompt or an LLM-generated response based on an expected set of characters. It is similar to traditional input validation, where certain characters are rejected, stripped, or escaped. For instance, SQL injection filters may escape single quotes, or HTML filters may strip left angular brackets from user input. However, LLM input queries and generated responses typically do not follow strict syntactic rules. Furthermore, LLM-based attack vectors such as prompt injection are challenging to prevent with character-based validation alone.

Let us implement a simple character-based validation example using Python's `Pydantic` library, a popular library used for data validation. We will use regex-based validators to ensure user input prompts and LLM-generated responses contain only whitelisted characters. For instance, we can implement an LLM-based calculator application that only accepts simple mathematical expressions containing the following characters:

- Digits: `0-9`
- Multiplication: `*`
- Division: `/`
- Addition: `+`
- Subtraction: `-`
- Brackets: `()`
- Decimal Point: `.`

We can use the following regular expression to implement this character whitelist:

Code: regex

```regex
^[0-9.\+\-\*/\(\)]+$
```

LLM-generated responses are only allowed to contain the result, i.e., digits and decimal points, which can be enforced using the following regular expression:

Code: regex

```regex
^[0-9\.]+$
```

Let us build a class to represent our LLM interaction, consisting of a user input prompt and a generated response, and enforce the regex validation using `Pydantic`:

Code: python

```python
from pydantic import BaseModel, StringConstraints
from typing import Annotated

class LLMQuery(BaseModel, validate_assignment=True):
	prompt: Annotated[str, StringConstraints(pattern=r"^[0-9.\+\-\*/\(\)]+$")]
	response: Annotated[str, StringConstraints(pattern=r"^[0-9\.]+$")] = None
```

To tell the LLM how it should behave, we can use the following system prompt:

Code: python

```python
SYSTEM_PROMPT = '''You are a calculator. Please compute the result of the following mathematical expression.
Only respond with the result, no other text.

'''
```

We will assume that we have a function `query_llm` that accepts a system prompt and a user input prompt as parameters and implements querying the LLM. The actual implementation may differ depending on the model and API used. Examples may include [openai](https://pypi.org/project/openai/) or [anthropic](https://pypi.org/project/anthropic/). In our example, we will assume that the function definition looks like this:

Code: python

```python
def query_llm(system_prompt:str, prompt: str) -> str:
	[...]
	return response
```

Using this implementation, we can now define a protected version that uses our validators to enforce character-based validation on user input and output:

Code: python

```python
def protected_query_llm(prompt: str) -> LLMQuery:
	query = LLMQuery(prompt=prompt)
	query.response = query_llm(SYSTEM_PROMPT, query.prompt)
	return query
```

Finally, we can test the validation using an endless loop:

Code: python

```python
while 1:
    try:
        prompt = input("> ")
        query_obj = protected_query_llm(prompt)
        print(query_obj.response)
    except Exception as e:
        print(f'Error: {e}')
```

We can submit some test inputs to test our implementation and see how the code reacts. While we can read the LLM-generated response to our input expression as long as we do not violate the imposed syntactical restrictions, the code throws an exception as soon as the validator is violated:

  Character-based Validation

```shell-session
@htb[/htb]$ python3 character_validator.py

> 7+7
14
> 2*2*2+3
11
> (3+3)*4
24
> 2.3*4       
9.2
> Hello World
Error: 1 validation error for LLMQuery
prompt
  String should match pattern '^[0-9.\+\-\*/\(\)]+$' [type=string_pattern_mismatch, input_value='Hello World', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/string_pattern_mismatch
```

While character-based validation is powerful in preventing traditional injection-based vulnerabilities such as SQL injection or Cross-Site Scripting, its usefulness is limited in an LLM-based context. For some targeted attacks, such as prompt injection, character-based validation is often ineffective because the respective prompt injection payloads often only consist of alphanumeric characters. Thus, limiting the type of special characters in the user input does not impede these attack vectors. On top of that, character-based validation can significantly affect the user experience. Since many LLM applications implement an interaction between users and the model based on free text without any restrictions, user input and LLM-generated responses need to contain special characters to convey relevant information. These special characters can include single quotes for quotation and angular brackets for equations or comparisons. Thus, limiting the character set for inputs or outputs often has only a limited mitigating effect while significantly reducing the user experience. Thus, most of the time, character-based validation is not a popular validator. However, there are use cases where it can be combined with other types of validation to prevent a variety of attack vectors effectively.

