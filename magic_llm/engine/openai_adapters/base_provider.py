import json
import logging
import mimetypes
import os
from abc import ABC
from typing import Dict, Tuple, Optional, Any

import aiohttp

from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse, ModelEmbeddingResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel
from magic_llm.util.http import AsyncHttpClient, HttpClient
from magic_llm.engine.tooling import map_request_tools

logger = logging.getLogger(__name__)


def _dump_payload(provider: "OpenAiBaseProvider", data: dict) -> str:
    """Grep-friendly JSON blob: model, messages summary, tools, tool_choice, base_url."""
    summary = {
        "marker": "MAGIC_LLM_DEBUG_PAYLOAD",
        "provider": provider.__class__.__name__,
        "model": getattr(provider, "model", None),
        "base_url": getattr(provider, "base_url", None),
    }
    msgs = data.get("messages", [])
    by_role: dict[str, int] = {}
    last_tool_call_names: list[str] = []
    tool_results: list[dict] = []
    for m in msgs:
        role = m.get("role", "?")
        by_role[role] = by_role.get(role, 0) + 1
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                last_tool_call_names.append(fn.get("name", "?"))
        if role == "tool":
            content = m.get("content", "")
            preview = (str(content)[:120] + "...") if content else "(empty)"
            tool_results.append({"tool_call_id": m.get("tool_call_id", "?"), "content_preview": preview})
    summary["messages"] = {"total": len(msgs), "by_role": by_role}
    if last_tool_call_names:
        summary["last_assistant_tool_calls"] = last_tool_call_names
    if tool_results:
        summary["tool_results"] = tool_results[:5]
    summary["tool_choice"] = data.get("tool_choice")
    tools_raw = data.get("tools")
    if tools_raw:
        summary["tools"] = [
            {"name": t.get("name", t.get("function", {}).get("name", "?")),
             "has_description": bool(t.get("description") or t.get("function", {}).get("description"))}
            for t in tools_raw
        ]
    if data.get("stream"):
        summary["stream"] = True
    return json.dumps(summary)


def _dump_payload_full(provider: "OpenAiBaseProvider", data: dict) -> str:
    """Full payload dump for diagnosis. Single-line JSON with stable marker."""
    payload = {
        "marker": "MAGIC_LLM_DEBUG_PAYLOAD_FULL",
        "provider": provider.__class__.__name__,
        "model": data.get("model"),
        "base_url": getattr(provider, "base_url", None),
        "stream": data.get("stream", False),
        "tool_choice": data.get("tool_choice"),
        "messages": data.get("messages"),
        "tools": data.get("tools"),
    }
    return json.dumps(payload, default=str)


def _has_image_content(messages: list[dict]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    return True
    return False


class OpenAiBaseProvider(ABC):
    supports_vision: bool = True
    
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model: str | None = None,
                 headers: Dict[str, str] = None,
                 **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        if headers:
            self.headers = headers
        else:
            self.headers = {}
        if 'user-agent' not in self.headers:
            self.headers['user-agent'] = 'arz-magic-llm-engine'
        if 'accept' not in self.headers:
            self.headers['accept'] = 'application/json'
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'
        if 'Authorization' not in self.headers:
            self.headers['Authorization'] = f'Bearer {api_key}'

        self.kwargs = kwargs
        self.model = model

    def prepare_data(self,
                     chat:
                     ModelChat,
                     **kwargs) -> Tuple[bytes, Dict[str, str]]:
        """
        Prepare the request data for OpenAI-compatible APIs.
        Alias for transform_request for backward compatibility.

        Note: Image handling is built into this method. OpenAI-compatible providers
        that support images will pass through the image_url content type.
        Providers that don't support images may need to override this method
        or handle the transformation in their specific implementation.
        """
        return self.transform_request(chat, **kwargs)

    def transform_request(self,
                          chat: ModelChat,
                          **kwargs) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to OpenAI-compatible request format.

        Args:
            chat: The input chat model
            **kwargs: Additional parameters (stream, tools, etc.)

        Returns:
            Tuple of (request_body_bytes, headers_dict)

        Raises:
            ChatException: If request contains images but provider doesn't support vision

        Note: Image support is provider-dependent. The base implementation
        passes through image_url content types. Providers that don't support
        images can override this method to filter or transform image content.
        """
        messages = chat.get_messages()
        
        # Strip non-standard is_error field from tool messages.
        # is_error is stored in ModelChat for internal debugging/tracing but is NOT
        # part of the OpenAI-compatible spec — strict providers may reject it.
        messages = [
            {k: v for k, v in msg.items() if k != 'is_error'}
            if msg.get('role') == 'tool' else msg
            for msg in messages
        ]

        if _has_image_content(messages) and not self.supports_vision:
            raise ChatException(
                message=f"Provider '{self.__class__.__name__}' does not support image/vision inputs. "
                        f"Model '{self.model}' cannot process images. Remove images or use a vision-capable provider.",
                error_code='VISION_NOT_SUPPORTED'
            )
        
        for message in messages:
            if message['role'] == 'user' and isinstance(message['content'], list):
                for item in message['content']:
                    if item.get('type') == 'text':
                        item.pop('image_url', None)
                    if item.get('type') == 'image_url':
                        item.pop('text', None)
        data = {
            "model": self.model,
            "messages": messages,
            **self.kwargs,
            **kwargs,
        }
        if 'callback' in data:
            data.pop('callback')
        if 'fallback' in data:
            data.pop('fallback')

        request_tools = map_request_tools('openai', data.get('tools'), data.get('tool_choice'))
        if request_tools.tools is not None:
            data['tools'] = request_tools.tools
        elif 'tools' in data:
            data.pop('tools')
        if request_tools.tool_choice is not None:
            data['tool_choice'] = request_tools.tool_choice
        elif 'tool_choice' in data:
            data.pop('tool_choice')

        if os.environ.get("MAGIC_LLM_DEBUG_PAYLOAD"):
            logger.info("MAGIC_LLM_DEBUG_PAYLOAD %s", _dump_payload(self, data))

        if os.environ.get("MAGIC_LLM_DEBUG_PAYLOAD_FULL"):
            logger.info("MAGIC_LLM_DEBUG_PAYLOAD_FULL %s", _dump_payload_full(self, data))

        json_data = json.dumps(data).encode('utf-8')
        return json_data, self.headers

    def transform_response(self, raw: Dict[str, Any]) -> ModelChatResponse:
        """
        Transform OpenAI API response to ModelChatResponse.
        OpenAI responses are already in the correct format.

        Args:
            raw: Raw response from OpenAI-compatible API

        Returns:
            Normalized ModelChatResponse
        """
        return ModelChatResponse(**raw)

    def transform_embedding_response(self, raw: Dict[str, Any]) -> ModelEmbeddingResponse:
        """
        Transform embedding API response to ModelEmbeddingResponse.

        Embedding responses follow the OpenAI format across all providers:
        - object: "list"
        - data: list of {object, index, embedding}
        - model: model identifier
        - usage: optional token usage

        Args:
            raw: Raw dict from embedding API response

        Returns:
            Normalized ModelEmbeddingResponse
        """
        return ModelEmbeddingResponse(**raw)

    def transform_stream_chunk(
        self,
        raw: str,
        context: Optional[Dict] = None
    ) -> Optional[ChatCompletionModel]:
        """
        Transform streaming chunk to ChatCompletionModel.
        Alias for process_chunk for backward compatibility.

        Args:
            raw: Raw chunk string from provider stream
            context: Optional context dict with 'id_generation' and 'last_chunk'

        Returns:
            Normalized ChatCompletionModel or None if chunk should be skipped
        """
        context = context or {}
        return self.process_chunk(
            raw,
            context.get('id_generation', ''),
            context.get('last_chunk')
        )

    def process_chunk(self,
                      chunk: str | dict,
                      id_generation: str = '',
                      last_chunk: ChatCompletionModel = None
                      ) -> ChatCompletionModel:
        if chunk.startswith('data: '):
            if '[DONE]' in chunk:
                return None
            chunk = json.loads(chunk[5:])
            # TODO improve server side error per provider
            if 'choices' not in chunk:
                raise Exception(f'no choices, {chunk}')
            chunk['usage'] = c if (c := chunk.get('usage', {})) else {}
            if len(chunk['choices']) == 0:
                return None
            chunk = ChatCompletionModel(**chunk)
            return chunk
        else:
            if chunk.strip():
                if not chunk.endswith('[DONE]') and not chunk.lower().startswith(': ping'):
                    return ChatCompletionModel(**{
                        'id': '0',
                        'model': 'dummy',
                        'choices': [
                            {
                                'delta': {
                                    'content': chunk
                                }
                            }
                        ]
                    })

    def prepare_async_transcriptions(self, data: AudioTranscriptionsRequest):
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            data.file,
            filename="audio.mp3",
            content_type=mimetypes.guess_type("audio.mp3")[0] or "application/octet-stream"
        )
        form_data.add_field('model', data.model)
        if data.language:
            form_data.add_field('language', data.language)
        if data.prompt:
            form_data.add_field('prompt', data.prompt)
        if data.response_format:
            form_data.add_field('response_format', data.response_format)
        if data.temperature is not None:
            form_data.add_field('temperature', str(data.temperature))
        return form_data

    def prepare_json_transcriptions(self, data: AudioTranscriptionsRequest):
        json_body = {
            "model": data.model,
        }

        # Add optional fields only if they exist
        if data.language:
            json_body["language"] = data.language
        if data.prompt:
            json_body["prompt"] = data.prompt
        if data.response_format:
            json_body["response_format"] = data.response_format
        if data.temperature is not None:
            json_body["temperature"] = data.temperature  # No need for str() conversion in JSON

        return json_body

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        raise NotImplementedError

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        headers = {
            "Authorization": self.headers.get("Authorization")
        }
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.base_url + '/audio/transcriptions',
                                              data=self.prepare_async_transcriptions(data),
                                              headers=headers)
            return response

    def sync_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        headers = {
            "Authorization": self.headers.get("Authorization")
        }
        with HttpClient() as client:
            response = client.post_json(url=self.base_url + '/audio/transcriptions',
                                        data=self.prepare_json_transcriptions(data),
                                        files={'file': ('audio.mp3', data.file, 'audio/mpeg')},
                                        headers=headers)
            return response
