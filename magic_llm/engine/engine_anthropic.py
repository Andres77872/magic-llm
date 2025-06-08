# https://docs.anthropic.com/claude/reference/messages-streaming
import json
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import (ChatCompletionModel,
                                             UsageModel,
                                             ChoiceModel,
                                             DeltaModel,
                                             PromptTokensDetailsModel)
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineAnthropic(BaseChat):
    engine = 'anthropic'

    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = 'https://api.anthropic.com/v1/messages'
        self.api_key = api_key

    def prepare_chunk(self, event: dict, idx, usage):
        if event['type'] == 'message_start':
            idx = event['message']['id']
            meta = event['message']['usage']
            usage = UsageModel(
                prompt_tokens=(
                        meta['input_tokens']
                        + meta.get('cache_read_input_tokens', 0)
                        + meta.get('cache_creation_input_tokens', 0)
                ),
                completion_tokens=meta['output_tokens'],
                total_tokens=(
                        meta['input_tokens']
                        + meta['output_tokens']
                        + meta.get('cache_read_input_tokens', 0)
                        + meta.get('cache_creation_input_tokens', 0)
                ),
                prompt_tokens_details=PromptTokensDetailsModel(cached_tokens=meta['cache_read_input_tokens']),
            )
            return None, idx, usage

        finish_reason = event.get('delta', {}).get('stop_reason', None)
        if event['type'] == 'message_delta':
            meta = event['usage']
            usage.completion_tokens = meta['output_tokens']
            usage.total_tokens += meta['output_tokens']
            return None, idx, usage
        if event['type'] == 'content_block_delta':
            delta = DeltaModel(content=event['delta']['text'], role=None)
            choice = ChoiceModel(delta=delta, finish_reason=finish_reason, index=0)
            model = ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage,
            )
            return model, idx, usage
        return None, idx, usage

    def prepare_data(self, chat: ModelChat, **kwargs):
        """
        Build the payload for Anthropic’s *Messages* endpoint.

        • Keeps the first `system` message (preamble) separate.
        • Converts every message’s `content` into the structure required by
          Anthropic:
              – plain strings  →  single text part
              – `image_url` parts
                    · data-URIs  →  in-line base-64     (source.type = base64)
                    · http/https →  referenced URL      (source.type = url)
          Other part types are passed through unchanged.
        • Adds `cache_control={"type": "ephemeral"}` to the text in the last
          three user turns.
        """
        # ------------------------------------------------------------------ #
        # HTTP headers
        # ------------------------------------------------------------------ #
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": (
                "messages-2023-12-15,"
                "pdfs-2024-09-25,"
                "prompt-caching-2024-07-31"
            ),
            "accept": "application/json",
            "user-agent": "arz-magic-llm-engine",
            **self.headers,
        }

        # ------------------------------------------------------------------ #
        # Extract optional pre-amble (1st `system` message)
        # ------------------------------------------------------------------ #
        messages = chat.get_messages()
        preamble = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
        if preamble:
            messages.pop(0)  # remove it from the normal flow

        # ------------------------------------------------------------------ #
        # Normalise message contents
        # ------------------------------------------------------------------ #
        anthropic_chat: list[dict] = []

        for msg in messages:
            # ----- simple string → single text part ----------------------- #
            if isinstance(msg["content"], str):
                parts = [{"type": "text", "text": msg["content"]}]

            # ----- list of parts ----------------------------------------- #
            else:
                parts = []
                for part in msg["content"]:
                    p_type = part.get("type")

                    # ----------------------------- text ------------------- #
                    if p_type == "text":
                        # strip helper keys that may be present
                        part.pop("image_url", None)
                        part.pop("document", None)
                        parts.append(part)

                    # ----------------------------- image ------------------ #
                    elif p_type == "image_url":
                        url: str = part["image_url"]["url"]

                        if url.startswith("data:"):
                            # data:<mime>;base64,<b64-data>
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(";")[0][5:]
                            source = {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            }
                        else:
                            # http / https – let Anthropic fetch the image
                            source = {
                                "type": "url",
                                "url": url,
                            }

                        parts.append({"type": "image", "source": source})

                    # ------------------------ any other part ------------- #
                    else:
                        parts.append(part)

            anthropic_chat.append({"role": msg["role"], "content": parts})

        # ------------------------------------------------------------------ #
        # Keep only the three most-recent *user* turns (with cache_control)
        # ------------------------------------------------------------------ #
        result: list[dict] = []
        user_turns_processed = 0

        for turn in reversed(anthropic_chat):
            if turn["role"] == "user" and user_turns_processed < 3:
                patched_parts = []
                for part in turn["content"]:
                    if part["type"] == "text":
                        patched_parts.append(
                            {
                                **part,
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                    else:
                        patched_parts.append(part)

                result.append({"role": "user", "content": patched_parts})
                user_turns_processed += 1
            else:
                result.append(turn)

        anthropic_chat = list(reversed(result))

        # ------------------------------------------------------------------ #
        # Final JSON body
        # ------------------------------------------------------------------ #
        data: dict = {
            "model": self.model,
            "messages": anthropic_chat,
            **kwargs,
            **self.kwargs,
        }
        data.setdefault("max_tokens", 4096)
        if preamble:
            data["system"] = preamble

        json_data = json.dumps(data).encode("utf-8")
        return json_data, headers

    def process_chunk(self, chunk: str, idx, usage):
        model, idx, usage = self.prepare_chunk(json.loads(chunk), idx, usage)
        return model, idx, usage

    def process_generate(self, r):
        return ModelChatResponse(**{
            'content': r['content'][0]['text'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=r['usage']['input_tokens'],
                completion_tokens=r['usage']['output_tokens'],
                total_tokens=r['usage']['input_tokens'] + r['usage']['output_tokens'],
            )
        })

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.base_url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))

            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.base_url,
                                        data=json_data,
                                        headers=headers)
            return self.process_generate(response)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        with HttpClient() as client:
            idx = None
            usage = None
            for chunk in client.stream_request("POST",
                                               self.base_url,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if chunk:
                    evt = chunk.split('data:')
                    if len(evt) != 2:
                        continue
                    if c := self.process_chunk(evt[-1].strip(), idx, usage):
                        idx = c[1]
                        usage = c[2]
                        if c[0]:
                            yield c[0]
            delta = DeltaModel(content='', role=None)
            choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
            yield ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage or UsageModel(),
            )

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        async with AsyncHttpClient() as client:
            idx = None
            usage = None
            async for chunk in client.post_stream(self.base_url,
                                                  data=json_data,
                                                  headers=headers):
                if chunk:
                    evt = chunk.decode().split('data:')
                    if len(evt) != 2:
                        continue
                    if c := self.process_chunk(evt[-1].strip(), idx, usage):
                        idx = c[1]
                    usage = c[2]
                    if c[0]:
                        yield c[0]
            delta = DeltaModel(content='', role=None)
            choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
            yield ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage or UsageModel(),
            )
