# https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=es-419
import asyncio
import base64
import io
import json
import logging
import mimetypes
import time
import wave
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall, Choice, Message
from magic_llm.model.ModelChatStream import (ChatCompletionModel,
                                             UsageModel,
                                             ChoiceModel,
                                             DeltaModel)
from magic_llm.util.http import AsyncHttpClient, HttpClient
from magic_llm.util.response_mapping import (
    GOOGLE_FINISH_REASON_MAP,
    map_finish_reason,
    build_response,
    build_stream_chunk,
    build_tool_call,
    build_stream_tool_call,
)

logger = logging.getLogger(__name__)


class EngineGoogle(BaseChat):
    engine = 'google'

    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        base = 'https://generativelanguage.googleapis.com/v1beta/models/'
        self.url_stream = f'{base}{self.model}:streamGenerateContent?alt=sse'
        self.url = f'{base}{self.model}:generateContent'
        self.url_tts = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent'
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
            "x-goog-api-key": self.api_key,
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

            # ---------- Already Gemini format (passthrough) -------------- #
            if "functionResponse" in part:
                return part
            if "functionCall" in part:
                return part
            if "inline_data" in part:
                return part
            if "executableCode" in part:
                return part
            if "code_execution_result" in part:
                return part

            raise ValueError(f"Unsupported part encountered: {part!r}")

        # ------------------------------------------------------------------ #
        # Convert every message to the Gemini "content" structure
        # ------------------------------------------------------------------ #
        with HttpClient() as client:
            api_contents: list[dict] = []
            for msg in messages:
                raw = msg["content"]

                if isinstance(raw, str):
                    parts = [{"text": raw}]
                elif isinstance(raw, list):
                    parts = [_convert_part(p, client) for p in raw]
                elif raw is None:
                    # Assistant messages with only tool_calls have content=None
                    parts = [{"text": ""}]
                else:
                    raise ValueError(f"Unknown content type: {type(raw)!r}")

                api_contents.append(
                    {
                        "role": msg["role"].replace("assistant", "model"),
                        "parts": parts,
                    }
                )

        # ----------------------- Final payload ---------------------------- #
        # Extract tools/tool_choice BEFORE building generationConfig (they are not JSON serializable)
        openai_tools = kwargs.pop('tools', None) or self.kwargs.get('tools', None)
        openai_tool_choice = kwargs.pop('tool_choice', None) or self.kwargs.get('tool_choice', None)

        data: dict = {
            "contents": api_contents,
            "generationConfig": {**self.kwargs, **kwargs},
        }
        if preamble:
            data["systemInstruction"] = {"parts": [{"text": preamble}]}

        if openai_tools is not None:
            # Detect already-serialized Gemini tools from GeminiToolAdapter
            # (format: [{"functionDeclarations": [...]}])
            if (isinstance(openai_tools, list) and len(openai_tools) == 1
                    and isinstance(openai_tools[0], dict)
                    and "functionDeclarations" in openai_tools[0]):
                # Already in Gemini format — pass through directly
                data["tools"] = openai_tools
                # Still map tool_choice if provided
                if openai_tool_choice is not None:
                    from magic_llm.util.tools_mapping import map_to_gemini
                    _, gemini_tool_config = map_to_gemini(None, openai_tool_choice)
                    if gemini_tool_config:
                        data["toolConfig"] = gemini_tool_config
            else:
                # OpenAI format or raw callables — convert via map_to_gemini
                from magic_llm.util.tools_mapping import map_to_gemini
                gemini_tools, gemini_tool_config = map_to_gemini(openai_tools, openai_tool_choice)
                if gemini_tools:
                    data["tools"] = [{"functionDeclarations": gemini_tools}]
                if gemini_tool_config:
                    data["toolConfig"] = gemini_tool_config

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
            "x-goog-api-key": self.api_key,
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

            # ---------- Already Gemini format (passthrough) -------------- #
            if "functionResponse" in part:
                return part
            if "functionCall" in part:
                return part
            if "inline_data" in part:
                return part
            if "executableCode" in part:
                return part
            if "code_execution_result" in part:
                return part

            raise ValueError(f"Unsupported part encountered: {part!r}")

        # ------------------------------------------------------------------ #
        # Convert every message to the Gemini "content" structure
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
                elif raw is None:
                    # Assistant messages with only tool_calls have content=None
                    parts = [{"text": ""}]
                else:
                    raise ValueError(f"Unknown content type: {type(raw)!r}")

                api_contents.append(
                    {
                        "role": msg["role"].replace("assistant", "model"),
                        "parts": parts,
                    }
                )

        # ----------------------- Final payload ---------------------------- #
        # Extract tools/tool_choice BEFORE building generationConfig (they are not JSON serializable)
        openai_tools = kwargs.pop('tools', None) or self.kwargs.get('tools', None)
        openai_tool_choice = kwargs.pop('tool_choice', None) or self.kwargs.get('tool_choice', None)

        data: dict = {
            "contents": api_contents,
            "generationConfig": {**self.kwargs, **kwargs},
        }
        if preamble:
            data["systemInstruction"] = {"parts": [{"text": preamble}]}

        if openai_tools is not None:
            # Detect already-serialized Gemini tools from GeminiToolAdapter
            # (format: [{"functionDeclarations": [...]}])
            if (isinstance(openai_tools, list) and len(openai_tools) == 1
                    and isinstance(openai_tools[0], dict)
                    and "functionDeclarations" in openai_tools[0]):
                # Already in Gemini format — pass through directly
                data["tools"] = openai_tools
                # Still map tool_choice if provided
                if openai_tool_choice is not None:
                    from magic_llm.util.tools_mapping import map_to_gemini
                    _, gemini_tool_config = map_to_gemini(None, openai_tool_choice)
                    if gemini_tool_config:
                        data["toolConfig"] = gemini_tool_config
            else:
                # OpenAI format or raw callables — convert via map_to_gemini
                from magic_llm.util.tools_mapping import map_to_gemini
                gemini_tools, gemini_tool_config = map_to_gemini(openai_tools, openai_tool_choice)
                if gemini_tools:
                    data["tools"] = [{"functionDeclarations": gemini_tools}]
                if gemini_tool_config:
                    data["toolConfig"] = gemini_tool_config

        json_bytes = json.dumps(data).encode("utf-8")
        return json_bytes, headers, data

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMATION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def transform_request(
        self,
        chat: ModelChat,
        **kwargs
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to Google Gemini request format.
        Note: This is a sync wrapper around prepare_data_sync.

        Image support: Google Gemini requires images to be inlined as base64.
        Remote images (HTTP/HTTPS) are automatically downloaded and embedded.
        """
        json_bytes, headers, _ = self.prepare_data_sync(chat, **kwargs)
        return json_bytes, headers

    async def async_transform_request(
        self,
        chat: ModelChat,
        **kwargs
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to Google Gemini request format (async version).
        Uses async HTTP client for downloading remote images.

        Image support: Google Gemini requires images to be inlined as base64.
        Remote images (HTTP/HTTPS) are automatically downloaded and embedded.
        """
        json_bytes, headers, _ = await self.prepare_data(chat, **kwargs)
        return json_bytes, headers

    def transform_response(self, raw: Dict[str, Any]) -> ModelChatResponse:
        """
        Transform Google Gemini response to ModelChatResponse.
        Uses shared response_mapping utilities.
        """
        return self.process_generate(raw)

    def transform_stream_chunk(
        self,
        raw: Any,
        context: Optional[Dict] = None
    ) -> Optional[ChatCompletionModel]:
        """
        Transform Google Gemini streaming chunk to ChatCompletionModel.

        Args:
            raw: Raw chunk string from Gemini stream
            context: Optional context dict (not used for Gemini)

        Returns:
            ChatCompletionModel
        """
        return self.prepare_stream_response(raw)

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
                func_call = part['functionCall']

                # Handle missing name field — skip with warning
                func_name = func_call.get('name')
                if not func_name:
                    logger.warning(
                        "Gemini functionCall part missing 'name' field, skipping: %s",
                        func_call,
                    )
                    continue

                if tool_calls is None:
                    tool_calls = []

                # Capture native id when available, fallback to synthetic ID
                call_id = func_call.get('id') or f"call_{len(tool_calls)}_{int(time.time() * 1000)}"

                tool_call = build_tool_call(
                    id=call_id,
                    name=func_name,
                    arguments=json.dumps(func_call.get('args', {}))
                )
                tool_calls.append(tool_call)

        # Combine text parts if any
        content = ''.join(content_parts) if content_parts else None

        # Map Gemini finish reasons to OpenAI format using shared mapping
        finish_reason = map_finish_reason(
            candidate.get('finishReason', 'STOP'),
            GOOGLE_FINISH_REASON_MAP,
            default=candidate.get('finishReason')
        )

        # Create usage model
        usage_metadata = gemini_response['usageMetadata']
        usage = UsageModel(
            prompt_tokens=usage_metadata['promptTokenCount'],
            completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
            total_tokens=usage_metadata['totalTokenCount']
        )

        # Build standardized response
        return build_response(
            id=gemini_response.get('responseId', f"gemini_{int(time.time() * 1000)}"),
            model=gemini_response.get('modelVersion', 'gemini'),
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
            logprobs=candidate.get('avgLogprobs')
        )

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = await self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout', 30))
            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = self.prepare_data_sync(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.url,
                                        data=json_data,
                                        headers=headers,
                                        timeout=kwargs.get('timeout', 30))
            return self.process_generate(response)

    def prepare_stream_response(self, chunk):
        payload = json.loads(chunk.strip()[5:].strip())
        usage = UsageModel(
            prompt_tokens=payload['usageMetadata']['promptTokenCount'],
            completion_tokens=payload['usageMetadata'].get('candidatesTokenCount', 0),
            total_tokens=payload['usageMetadata']['totalTokenCount'],
        )

        candidate = payload['candidates'][0]
        parts = candidate['content']['parts']

        # Extract text and tool calls from all parts
        content = None
        tool_calls = None

        for part in parts:
            if 'text' in part:
                content = part['text']
            elif 'functionCall' in part:
                # AI Studio: full functionCall arrives atomically (no partial args)
                func_call = part['functionCall']
                func_name = func_call.get('name')
                if not func_name:
                    logger.warning(
                        "Streaming functionCall part missing 'name' field, skipping: %s",
                        func_call,
                    )
                    continue
                if tool_calls is None:
                    tool_calls = []
                tool_call = build_stream_tool_call(
                    id=func_call.get('id', f"call_{len(tool_calls)}_{int(time.time() * 1000)}"),
                    name=func_name,
                    arguments=json.dumps(func_call.get('args', {}))
                )
                tool_calls.append(tool_call)

        return build_stream_chunk(
            id='1',
            model=self.model,
            content=content or '',
            role='assistant',
            tool_calls=tool_calls,
            usage=usage
        )

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers, data = self.prepare_data_sync(chat, **kwargs)

        with HttpClient() as client:
            for chunk in client.stream_request("POST",
                                               self.url_stream,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout', 30)):
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

    def _pcm_to_wav_bytes(self, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
        """Convert PCM data to WAV format and return as bytes."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
        wav_buffer.seek(0)
        return wav_buffer.read()

    def _prepare_tts_data(self, speech_request: AudioSpeechRequest) -> dict:
        """Prepare data for Gemini TTS API request."""
        return {
            "contents": [{
                "parts": [{
                    "text": speech_request.input
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": speech_request.voice
                        }
                    }
                }
            },
            "model": speech_request.model,
        }

    def audio_speech(self, speech_request: AudioSpeechRequest, **kwargs) -> bytes:
        """Generate audio speech synchronously using Gemini TTS."""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
            **self.headers
        }

        data = self._prepare_tts_data(speech_request)
        json_data = json.dumps(data).encode("utf-8")

        with HttpClient() as client:
            response = client.post_json(
                url=self.url_tts,
                data=json_data,
                headers=headers,
                timeout=kwargs.get('timeout', 30)
            )

            # Extract PCM audio data from response
            pcm_data = base64.b64decode(response['candidates'][0]['content']['parts'][0]['inlineData']['data'])

            # Convert PCM to WAV bytes and return
            return self._pcm_to_wav_bytes(pcm_data)

    async def async_audio_speech(self, speech_request: AudioSpeechRequest, **kwargs) -> bytes:
        """Generate audio speech asynchronously using Gemini TTS."""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
            **self.headers
        }

        data = self._prepare_tts_data(speech_request)
        json_data = json.dumps(data).encode("utf-8")

        async with AsyncHttpClient() as client:
            response = await client.post_json(
                url=self.url_tts,
                data=json_data,
                headers=headers,
                timeout=kwargs.get('timeout', 30)
            )

            # Extract PCM audio data from response
            pcm_data = base64.b64decode(response['candidates'][0]['content']['parts'][0]['inlineData']['data'])

            # Convert PCM to WAV bytes and return
            return self._pcm_to_wav_bytes(pcm_data)
