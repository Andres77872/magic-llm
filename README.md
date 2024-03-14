# Magic LLM

Magic LLM is a simplified wrapper designed to facilitate connections with various LLM providers, including:

- [OpenAI](https://platform.openai.com/docs/api-reference)
- [Cloudflare](https://developers.cloudflare.com/workers-ai/models/text-generation/#responses)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Google AI Studio](https://ai.google.dev/tutorials/rest_quickstart)
- [Cohere](https://docs.cohere.com/reference/chat) (Non stream not supported)
- [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api) (Non stream not supported)

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

Note: Other LLM providers have not been tested.

## Features

- [x] Chat completion
- [ ] Text completion (Currently not supported)
- [ ] Embedding
  - [x] OpenAI
  - [x] Together.AI
  - [x] Anyscale
  - [ ] DeepInfra
  - [x] Fireworks.AI
  - [x] Mistral
  - [ ] Cloudflare
  - [ ] Cohere
- [x] No stream response
- [x] Stream response reforged to map to OpenAI response format
- [x] Function calling (Only tested with OpenAI)

## Purpose

This client is not intended to replace the full functionality of the OpenAI client. Instead, it has been developed as the core component for another project, [Magic UI](https://magic-ui.arz.ai/), which is currently under development. The goal of Magic UI is to create a robust application generator (RAG).

# Clients

This client is built to be compatible with OpenAI's client, aiming to unify multiple LLM providers under the same framework.

## OpenAI and any other with the same API compatibility

```python
client = MagicLLM(
    engine='openai',
    model='gpt-3.5-turbo-0125',
    private_key='sk-',
    stream=True
)
```

## Cloudflare

```python
client = MagicLLM(
    engine='cloudflare',
    model='@cf/mistral/mistral-7b-instruct-v0.1',
    private_key='a...b',
    account_id='c...1',
    stream=True
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
    stream=True
)
```

## Google AI studio

```python
client = MagicLLM(
    engine='google',
    model='gemini-pro',
    private_key='A...B',
    stream=True
)
```

## Cohere

```python
client = MagicLLM(
    engine='cohere',
    model='command-light',
    private_key='a...b',
    stream=True
)
```

## Usage (same for all clients)

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

client = MagicLLM(
    engine='openai',
    model='gpt-3.5-turbo-0125',
    private_key='sk-',
    stream=True
)

chat = ModelChat(system="You are an assistant who responds sarcastically.")
chat.add_user_message("Hello, my name is Andres.")
chat.add_assistant_message("What an original name. 🙄")
chat.add_user_message("Thanks, you're also as original as an ant in an anthill.")

for i in client.llm.stream_generate(chat):
    print(i)
```