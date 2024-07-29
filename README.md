# Magic LLM

Magic LLM is a simplified wrapper designed to facilitate connections with various LLM providers, including:

- [OpenAI](https://platform.openai.com/docs/api-reference)
- [Cloudflare](https://developers.cloudflare.com/workers-ai/models/text-generation/#responses)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Google AI Studio](https://ai.google.dev/tutorials/rest_quickstart)
- [Cohere](https://docs.cohere.com/reference/chat)
- [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

## Tested LLM Providers with OpenAI Compatibility

The following LLM providers have been tested for compatibility with OpenAI's API:

- [Perplexity AI](https://docs.perplexity.ai/reference/post_chat_completions)
- [Together.AI](https://docs.together.ai/docs/openai-api-compatibility)
- [Anyscale](https://docs.endpoints.anyscale.com/examples/work-with-openai)
- [OpenRouter](https://openrouter.ai/docs#requests)
- [DeepInfra](https://deepinfra.com/docs/advanced/openai_api)
- [Fireworks.AI](https://readme.fireworks.ai/reference/createchatcompletion)
- [Mistral](https://docs.mistral.ai/api/#operation/createChatCompletion)
- [Deepseek](https://platform.deepseek.com/docs)
- [Groq](https://console.groq.com/docs/openai)
- [LeptonAI](https://www.lepton.ai/docs/public_models/model_apis)
- [OctoAI](https://octo.ai/docs/text-gen-solution/rest-api)
- [NovitaAI](https://novita.ai/get-started/llm.html)

Note: Other LLM providers have not been tested.

## Features

- [x] Chat completion
- [x] Text completion
- [x] Embedding
- [x] No stream response
- [x] Stream response reforged to map to OpenAI response format
- [x] Function calling (Only tested with OpenAI)
- [x] Stream yields a [chunk object](https://platform.openai.com/docs/api-reference/chat/streaming)
- [x] Usage in response
- [ ] Vision adapter to OpenAI schema
  - [x] OpenAI
  - [x] Anthropic
  - [ ] Google AI Studio
- [ ] Text to Speech
  - [x] OpenAI
- [ ] Fallback client
  - [x] Stream completion
  - [ ] completion

| provider         | Streaming | completion | embedding | async streaming | async completion | async embedding |
|------------------|-----------|------------|-----------|-----------------|------------------|-----------------|
| OpenAI           | ✅         | ✅          | ✅         | ✅               | ✅                | ❌               |
| Cloudflare       | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| AWS Bedrock      | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Google AI Studio | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Cohere           | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Anthropic        | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Perplexity AI    | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Together.AI      | ✅         | ✅          | ✅         | ✅               | ✅                | ❌               |
| OpenRouter       | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| DeepInfra        | ✅         | ✅          | ✅         | ✅               | ✅                | ❌               |
| Fireworks.AI     | ✅         | ✅          | ✅         | ✅               | ✅                | ❌               |
| Mistral          | ✅         | ✅          | ✅         | ✅               | ✅                | ❌               |
| Deepseek         | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| Groq             | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| LeptonAI         | ✅         | ✅          | ❌         | ⁉               | ✅                | ❌               |
| OctoAI           | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |
| NovitaAI         | ✅         | ✅          | ❌         | ✅               | ✅                | ❌               |

## Purpose

This client is not intended to replace the full functionality of the OpenAI client. Instead, it has been developed as
the core component for another project, [Magic UI](https://talk.novus.chat/), which is currently under development. The
goal of Magic UI is to create a robust application generator (RAG).

# Clients

This client is built to be compatible with OpenAI's client, aiming to unify multiple LLM providers under the same
framework.

## OpenAI and any other with the same API compatibility

```python
client = MagicLLM(
    engine='openai',
    model='gpt-3.5-turbo-0125',
    private_key='sk-',
)
```

## Cloudflare

```python
client = MagicLLM(
    engine='cloudflare',
    model='@cf/mistral/mistral-7b-instruct-v0.1',
    private_key='a...b',
    account_id='c...1',
)
```

## Amazon bedrock

```python
client = MagicLLM(
    engine='amazon',
    model='amazon.titan-text-express-v1',
    aws_access_key_id='A...B',
    aws_secret_access_key='a...b',
    region_name='us-east-1',
)
```

## Google AI studio

```python
client = MagicLLM(
    engine='google',
    model='gemini-pro',
    private_key='A...B',
)
```

## Cohere

```python
client = MagicLLM(
    engine='cohere',
    model='command-light',
    private_key='a...b',
)
```

## Usage (same for all clients)

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

client_fallback = MagicLLM(
    engine='openai',
    model='gpt-3.5-turbo-0125',
    private_key='sk-',
    # base_url='API'
)

client = MagicLLM(
    engine='openai',
    model='model_fail',
    private_key='sk-',
    # base_url='API',
    fallback=client_fallback
)

chat = ModelChat(system="You are an assistant who responds sarcastically.")
chat.add_user_message("Hello, my name is Andres.")
chat.add_assistant_message("What an original name. 🙄")
chat.add_user_message("Thanks, you're also as original as an ant in an anthill.")

for i in client.llm.stream_generate(chat):
    print(i)
```