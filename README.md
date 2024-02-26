# Magic LLM

Magic LLM is a simplified wrapper designed to facilitate connections with various LLM providers, including:

- [OpenAI](https://platform.openai.com/docs/api-reference)
- [Cloudflare](https://developers.cloudflare.com/workers-ai/models/text-generation/#responses)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Google AI Studio](https://ai.google.dev/tutorials/rest_quickstart)

## Tested LLM Providers with OpenAI Compatibility

The following LLM providers have been tested for compatibility with OpenAI's API:

- [Perplexity AI](https://docs.perplexity.ai/reference/post_chat_completions)
- [Together.AI](https://docs.together.ai/docs/openai-api-compatibility)
- [Anyscale](https://docs.endpoints.anyscale.com/examples/work-with-openai)
- [OpenRouter](https://openrouter.ai/docs#requests)
- [DeepInfra](https://deepinfra.com/docs/advanced/openai_api)
- [Fireworks.AI](https://readme.fireworks.ai/reference/createchatcompletion)

Note: Other LLM providers have not been tested.

## Features

- [x] Chat completion
- [ ] Text completion (Currently not supported)
- [x] No stream response
- [x] Stream response reforged to map to OpenAI response format
- [x] Function calling (Only tested with OpenAI)

## Purpose

This client is not intended to replace the full functionality of the OpenAI client. Instead, it has been developed as the core component for another project, [Magic UI](https://magic-ui.arz.ai/), which is currently under construction. The goal of Magic UI is to build a robust application generator (RAG).
