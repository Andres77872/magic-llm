# https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=es-419
import asyncio
import base64
import json
import mimetypes
import time
from urllib.parse import urlparse

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall, Choice, Message
from magic_llm.model.ModelChatStream import (ChatCompletionModel,
                                             UsageModel,
                                             ChoiceModel,
                                             DeltaModel)
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineGoogle(BaseChat):
    engine = 'google'

    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        base = 'https://generativelanguage.googleapis.com/v1beta/models/'
        self.url_stream = f'{base}{self.model}:streamGenerateContent?alt=sse&key={api_key}'
        self.url = f'{base}{self.model}:generateContent?key={api_key}'
        self.api_key = api_key

    def prepare_data_sync(self, chat: ModelChat, **kwargs):
        """
        Synchronous counterpart of `prepare_data` that in-lines remote images
        using the *HttpClient* helper (requests-based).

        A 5-MB size cap and a 5-second timeout are enforced for every download
        to minimise the risk of blocking the server for too long.

        Returns
        -------
        tuple[bytes, dict, dict]
            (json_bytes, http_headers, python_dict_payload)
        """
        # ------------------------ Tunables -------------------------------- #
        MAX_BYTES = 5 * 1024 * 1024  # hard cap per image
        TIMEOUT = 5  # seconds

        # ------------------------- Headers -------------------------------- #
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self.headers,
        }

        # ------------------------- Messages ------------------------------- #
        messages = chat.get_messages().copy()

        preamble: str | None = None
        if messages and messages[0]["role"] == "system":
            preamble = messages.pop(0)["content"]

        # ------------------ Sync helpers ---------------------------------- #
        def _http_image_to_b64(url: str, client: HttpClient) -> tuple[str, str]:
            """
            Download an image (< MAX_BYTES) and return (mime_type, b64_str).
            """
            content = client.request("GET", url, timeout=TIMEOUT)
            if len(content) > MAX_BYTES:
                raise ValueError(f"Image from {url!r} exceeds {MAX_BYTES // 1024} KB")

            mime = mimetypes.guess_type(url)[0] or "application/octet-stream"
            return mime, base64.b64encode(content).decode("utf-8")

        def _convert_part(part: dict, client: HttpClient) -> dict:
            kind = part.get("type")

            # ---------- Text --------------------------------------------- #
            if kind == "text":
                return {"text": part["text"]}

            # ---------- Image -------------------------------------------- #
            if kind == "image_url":
                url: str = part["image_url"]["url"]

                # Already an inline data-URI
                if url.startswith("data:"):
                    mime, b64 = url.split(";")[0][5:], url.split(",")[1]
                    return {"inline_data": {"mime_type": mime, "data": b64}}

                # Remote HTTP/S ─ fetch & inline
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"}:
                    mime, b64 = _http_image_to_b64(url, client)
                    return {"inline_data": {"mime_type": mime, "data": b64}}

            raise ValueError(f"Unsupported part encountered: {part!r}")

        # ------------------------------------------------------------------ #
        # Convert every message to the Gemini “content” structure
        # ------------------------------------------------------------------ #
        with HttpClient() as client:
            api_contents: list[dict] = []
            for msg in messages:
                raw = msg["content"]

                if isinstance(raw, str):
                    parts = [{"text": raw}]
                elif isinstance(raw, list):
                    parts = [_convert_part(p, client) for p in raw]
                else:
                    raise ValueError(f"Unknown content type: {type(raw)!r}")

                api_contents.append(
                    {
                        "role": msg["role"].replace("assistant", "model"),
                        "parts": parts,
                    }
                )

        # ----------------------- Final payload ---------------------------- #
        data: dict = {
            "contents": api_contents,
            "generationConfig": {**self.kwargs, **kwargs},
        }
        if preamble:
            data["systemInstruction"] = {"parts": [{"text": preamble}]}

        json_bytes = json.dumps(data).encode("utf-8")
        return json_bytes, headers, data

    async def prepare_data(self, chat: ModelChat, **kwargs):
        """
        Build the request body and headers for the Gemini API.

        Images referenced through HTTP/S URLs are downloaded asynchronously
        (5-MB cap, 5-second timeout) and then embedded as base-64 `inline_data`
        so the payload is 100 % self-contained.

        Returns
        -------
        tuple[bytes, dict, dict]
            (json_bytes, http_headers, python_dict_payload)
        """
        # ------------------------ Tunables -------------------------------- #
        MAX_BYTES = 5 * 1024 * 1024  # hard cap per image
        TIMEOUT = 5  # seconds

        # ------------------------- Headers -------------------------------- #
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self.headers,
        }

        # ------------------------- Messages ------------------------------- #
        messages = chat.get_messages().copy()

        preamble: str | None = None
        if messages and messages[0]["role"] == "system":
            preamble = messages.pop(0)["content"]

        # ------------------ Async helpers --------------------------------- #
        async def _http_image_to_b64(url: str, client: AsyncHttpClient) -> tuple[str, str]:
            """
            Download an image (< MAX_BYTES) and return (mime_type, b64_str).
            """
            content = await client.request("GET", url, timeout=TIMEOUT)
            if len(content) > MAX_BYTES:
                raise ValueError(f"Image from {url!r} exceeds {MAX_BYTES // 1024} KB")

            mime = mimetypes.guess_type(url)[0] or "application/octet-stream"
            return mime, base64.b64encode(content).decode("utf-8")

        async def _convert_part(part: dict, client: AsyncHttpClient) -> dict:
            kind = part.get("type")

            # ---------- Text --------------------------------------------- #
            if kind == "text":
                return {"text": part["text"]}

            # ---------- Image -------------------------------------------- #
            if kind == "image_url":
                url: str = part["image_url"]["url"]

                # Already an inline data-URI
                if url.startswith("data:"):
                    mime, b64 = url.split(";")[0][5:], url.split(",")[1]
                    return {"inline_data": {"mime_type": mime, "data": b64}}

                # Remote HTTP/S ─ fetch & inline
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"}:
                    mime, b64 = await _http_image_to_b64(url, client)
                    return {"inline_data": {"mime_type": mime, "data": b64}}

            raise ValueError(f"Unsupported part encountered: {part!r}")

        # ------------------------------------------------------------------ #
        # Convert every message to the Gemini “content” structure
        # ------------------------------------------------------------------ #
        async with AsyncHttpClient() as client:
            api_contents: list[dict] = []
            for msg in messages:
                raw = msg["content"]

                if isinstance(raw, str):
                    parts = [{"text": raw}]
                elif isinstance(raw, list):
                    parts_tasks = [_convert_part(p, client) for p in raw]
                    parts = await asyncio.gather(*parts_tasks)
                else:
                    raise ValueError(f"Unknown content type: {type(raw)!r}")

                api_contents.append(
                    {
                        "role": msg["role"].replace("assistant", "model"),
                        "parts": parts,
                    }
                )

        # ----------------------- Final payload ---------------------------- #
        data: dict = {
            "contents": api_contents,
            "generationConfig": {**self.kwargs, **kwargs},
        }
        if preamble:
            data["systemInstruction"] = {"parts": [{"text": preamble}]}

        json_bytes = json.dumps(data).encode("utf-8")
        return json_bytes, headers, data

    def process_generate(self, gemini_response: dict) -> ModelChatResponse:
        """Convert Gemini API response to ModelChatResponse format"""

        # Extract content and tool calls from parts
        content_parts = []
        tool_calls = None

        candidate = gemini_response['candidates'][0]
        parts = candidate['content']['parts']

        for part in parts:
            if 'text' in part:
                # Text content
                content_parts.append(part['text'])
            elif 'functionCall' in part:
                # Function/tool call
                if tool_calls is None:
                    tool_calls = []

                func_call = part['functionCall']
                tool_call = (ToolCall(
                    id=f"call_{len(tool_calls)}_{int(time.time() * 1000)}",  # Generate unique ID
                    type='function',
                    function=FunctionCall(
                        name=func_call['name'],
                        arguments=json.dumps(func_call.get('args', {}))
                    )
                ))
                tool_calls.append(tool_call)

        # Combine text parts if any
        content = ''.join(content_parts) if content_parts else None

        # Create the message
        message = Message(
            role='assistant',  # Gemini uses 'model' but we normalize to 'assistant'
            content=content,
            tool_calls=tool_calls,
            refusal=None,
            annotations=[]
        )

        # Map Gemini finish reasons to OpenAI format
        finish_reason_map = {
            'STOP': 'stop',
            'MAX_TOKENS': 'length',
            'SAFETY': 'content_filter',
            'RECITATION': 'content_filter',
            'OTHER': 'stop',
            'FINISH_REASON_UNSPECIFIED': None
        }
        finish_reason = finish_reason_map.get(
            candidate.get('finishReason', 'STOP'),
            candidate.get('finishReason')
        )

        # Create the choice
        choice = Choice(
            index=0,
            message=message,
            logprobs=candidate.get('avgLogprobs'),  # Gemini provides average log probs
            finish_reason=finish_reason
        )

        # Create usage model
        usage_metadata = gemini_response['usageMetadata']
        usage = UsageModel(
            prompt_tokens=usage_metadata['promptTokenCount'],
            completion_tokens=usage_metadata['candidatesTokenCount'],
            total_tokens=usage_metadata['totalTokenCount']
        )

        # Create the final response
        response = ModelChatResponse(
            id=gemini_response.get('responseId', f"gemini_{int(time.time() * 1000)}"),
            object='chat.completion',
            created=int(time.time()),  # Gemini doesn't provide timestamp
            model=gemini_response.get('modelVersion', 'gemini'),
            choices=[choice],
            usage=usage,
            service_tier=None,  # Gemini doesn't provide this
            system_fingerprint=None  # Gemini doesn't provide this
        )

        return response

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = await self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))
            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = self.prepare_data_sync(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.url,
                                        data=json_data,
                                        headers=headers)
            return self.process_generate(response)

    def prepare_stream_response(self, chunk):
        payload = json.loads(chunk.strip()[5:].strip())
        usage = UsageModel(
            prompt_tokens=payload['usageMetadata']['promptTokenCount'],
            completion_tokens=payload['usageMetadata'].get('candidatesTokenCount', 0),
            total_tokens=payload['usageMetadata']['totalTokenCount'],
        )
        delta = DeltaModel(content=payload['candidates'][0]['content']['parts'][0]['text'], role='assistant')
        choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
        return ChatCompletionModel(
            id='1',
            choices=[choice],
            created=int(time.time()),
            model=self.model,
            object='chat.completion.chunk',
            usage=usage,
        )

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers, data = self.prepare_data_sync(chat, **kwargs)

        with HttpClient() as client:
            for chunk in client.stream_request("POST",
                                               self.url_stream,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if chunk.strip():
                    yield self.prepare_stream_response(chunk)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers, _ = await self.prepare_data(chat, **kwargs)

        async with AsyncHttpClient() as client:
            async for chunk in client.post_stream(self.url_stream,
                                                  data=json_data,
                                                  headers=headers):
                if chunk.strip():
                    yield self.prepare_stream_response(chunk)
