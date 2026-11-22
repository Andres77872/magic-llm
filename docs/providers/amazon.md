# Amazon Bedrock

Use `engine="amazon"` for AWS Bedrock.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="amazon",
    model="amazon.nova-pro-v1:0",
    private_key="AKIA-your-access-key-id",
    aws_secret_access_key="your-secret-access-key",
    region_name="us-east-1",
)
```

`private_key` is used as the AWS access key ID by the base constructor. Some existing examples may pass `aws_access_key_id`; prefer the shape above unless your wrapper maps that value into `private_key`. `region_name` defaults to `us-east-1`.

If either key is omitted, credentials come from botocore's ambient chain: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, or an IAM role. Session tokens from that chain are included in signing.

## Model routing

The Bedrock model ID prefix selects the request format:

| Model ID prefix | Adapter |
| --- | --- |
| `amazon.nova...` or `arn:aws:...` | Nova |
| other `amazon...` | Titan |
| `anthropic...` | Anthropic on Bedrock |
| `meta...` | Meta Llama |

Any other prefix raises `ValueError("Unsupported model: ...")` at construction.

## Notes

- Chat, streaming, async chat, and async streaming are supported (`/invoke` and `/invoke-with-response-stream`).
- Amazon Polly sync TTS is supported through `client.llm.audio_speech(...)`.
- Async TTS and all STT methods are unsupported; Amazon Transcribe is not implemented in core.
- Vision input is disabled for every Bedrock adapter (Nova, Titan, Anthropic, Meta). Images are rejected before any request is sent; see [unsupported-feature errors](index.md#unsupported-feature-errors).
- Tool calling is not wired for the Amazon engine. Passing `tools` raises `TOOL_WIRING_NOT_IMPLEMENTED`.
- Embeddings are not implemented: `embedding()` and `async_embedding()` return `None`.
- Model discovery is unsupported and raises `NotImplementedError`.
- Requests use AWS Signature Version 4 through `botocore`-compatible signing utilities.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="You are an AWS assistant.")
chat.add_user_message("Explain Bedrock model IDs in one paragraph.")

response = client.llm.generate(chat)
print(response.content)
```
