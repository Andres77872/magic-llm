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

`private_key` is used as the AWS access key ID by the base constructor. Some existing examples may pass `aws_access_key_id`; prefer the shape above unless your wrapper maps that value into `private_key`.

## Notes

- Chat, streaming, async chat, and async streaming are supported.
- Amazon Polly sync TTS is supported through `client.llm.audio_speech(...)`.
- Async TTS and all STT methods are unsupported; Amazon Transcribe is not implemented in core.
- Model routing is based on the Bedrock model prefix, for example Nova, Titan, Anthropic, or Meta model IDs.
- Amazon Nova vision input is disabled until a native Bedrock image payload transform is implemented and tested.
- Tool calling is not wired for the Amazon engine.
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
