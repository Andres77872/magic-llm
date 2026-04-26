# https://docs.anthropic.com/claude/reference/messages-streaming
import json
import time
from typing import Dict, Any, Tuple, Optional

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import (
    ToolCall as RespToolCall,
    FunctionCall as RespFunctionCall,
    Message,
    Choice,
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    UsageModel,
    ChoiceModel,
    DeltaModel,
    PromptTokensDetailsModel,
    ToolCall as StreamToolCall,
    FunctionCall as StreamFunctionCall,
)
from magic_llm.util.http import AsyncHttpClient, HttpClient
from magic_llm.util.tools_mapping import map_to_anthropic
from magic_llm.util.response_mapping import (
    ANTHROPIC_FINISH_REASON_MAP,
    map_finish_reason,
    build_response,
    build_stream_chunk,
    build_tool_call,
    build_stream_tool_call,
)


class EngineAnthropic(BaseChat):
    engine = 'anthropic'

    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = 'https://api.anthropic.com/v1/messages'
        self.api_key = api_key
        # Streaming state: tracks active content blocks (by event index)
        # so we can assemble tool_use partial JSON and map to OpenAI-style tool_calls
        self._active_blocks: dict[int, dict] = {}
        # Persist the latest mapped finish_reason (OpenAI-style) for the stream
        self._finish_reason: str | None = None

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
                prompt_tokens_details=PromptTokensDetailsModel(cached_tokens=meta.get('cache_read_input_tokens', 0)),
            )
            return None, idx, usage

        # Map Anthropic stop_reason to OpenAI-style finish_reason and persist it
        finish_reason = event.get('delta', {}).get('stop_reason', None)
        if finish_reason:
            self._finish_reason = map_finish_reason(
                finish_reason,
                ANTHROPIC_FINISH_REASON_MAP,
                default=finish_reason
            )

        # Debug helper (disabled to avoid noisy output)
        # print('event:', event)

        if event['type'] == 'message_delta':
            meta = event['usage']
            usage.completion_tokens = meta['output_tokens']
            # message_delta output_tokens is typically cumulative; recompute total
            usage.total_tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            return None, idx, usage
        # Capture final stop reason if provided on message_stop events
        if event['type'] == 'message_stop':
            sr = event.get('message', {}).get('stop_reason') or event.get('stop_reason')
            if sr:
                self._finish_reason = map_finish_reason(
                    sr,
                    ANTHROPIC_FINISH_REASON_MAP,
                    default=sr
                )
            return None, idx, usage
        # Track start/stop of content blocks to properly handle tool_use
        if event['type'] == 'content_block_start':
            # Save the content block context by its stream index
            # Example for tool use: {"type":"tool_use","id":"toolu_...","name":"get_weather","input":{}}
            self._active_blocks[event['index']] = {
                'block': event.get('content_block', {}),
                'args': ''  # accumulate partial_json for tool_use
            }
            return None, idx, usage
        if event['type'] == 'content_block_stop':
            # Cleanup finished block
            self._active_blocks.pop(event['index'], None)
            return None, idx, usage
        if event['type'] == 'content_block_delta':
            d = event.get('delta', {})
            d_type = d.get('type')

            # Text delta (normal content)
            if 'text' in d:
                model = build_stream_chunk(
                    id=idx,
                    model=self.model,
                    content=d.get('text', ''),
                    finish_reason=finish_reason,
                    usage=usage
                )
                return model, idx, usage

            # Tool use JSON input streaming
            if d_type == 'input_json_delta':
                block_ctx = self._active_blocks.get(event['index'], {})
                block = block_ctx.get('block', {})
                if block.get('type') == 'tool_use':
                    # Append partial JSON chunk to accumulated args string
                    partial = d.get('partial_json', '') or ''
                    block_ctx['args'] = block_ctx.get('args', '') + partial
                    self._active_blocks[event['index']] = block_ctx

                    tool_call = build_stream_tool_call(
                        id=block.get('id'),
                        name=block.get('name'),
                        arguments=block_ctx['args']
                    )
                    model = build_stream_chunk(
                        id=idx,
                        model=self.model,
                        content='',
                        finish_reason=finish_reason,
                        tool_calls=[tool_call],
                        usage=usage
                    )
                    return model, idx, usage

            # Unknown delta type: ignore gracefully
            return None, idx, usage
        return None, idx, usage

    def prepare_data(self, chat: ModelChat, **kwargs):
        """
        Build the payload for Anthropic's *Messages* endpoint.

        • Keeps the first `system` message (preamble) separate.
        • Converts every message's `content` into the structure required by
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
        use_cache = kwargs.pop('use_cache', True)
        stream_mode = kwargs.pop('stream', False)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "accept": "text/event-stream" if stream_mode else "application/json",
            "user-agent": "arz-magic-llm-engine",
            **self.headers,
        }

        if use_cache:
            headers["anthropic-beta"] = (
                "messages-2023-12-15,"
                "pdfs-2024-09-25,"
                "prompt-caching-2024-07-31"
            )
        else:
            headers["anthropic-beta"] = (
                "messages-2023-12-15,"
                "pdfs-2024-09-25"
            )

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
            if turn["role"] == "user" and user_turns_processed < 3 and use_cache:
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
        # Support OpenAI-style functions/tools and tool_choice via unified mapper
        # Respect call-time overrides even when falsy (e.g., [], None)
        if 'tools' in kwargs:
            openai_tools = kwargs.pop('tools')
        else:
            openai_tools = self.kwargs.get('tools', None)
        if 'tool_choice' in kwargs:
            openai_tool_choice = kwargs.pop('tool_choice')
        else:
            openai_tool_choice = self.kwargs.get('tool_choice', None)

        data: dict = {
            'model': self.model,
            'messages': anthropic_chat,
            # init-time defaults first, call-time overrides second
            **self.kwargs,
            **kwargs,
        }

        # Enable server-sent events when streaming
        if stream_mode:
            data['stream'] = True

        anthropic_tools, anthropic_choice = map_to_anthropic(openai_tools, openai_tool_choice)
        if anthropic_tools:
            data['tools'] = anthropic_tools
        if anthropic_choice:
            data['tool_choice'] = anthropic_choice

        data.setdefault('max_tokens', 4096)
        if preamble:
            data['system'] = preamble

        json_data = json.dumps(data).encode('utf-8')
        return json_data, headers

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMATION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def transform_request(
        self,
        chat: ModelChat,
        **kwargs
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to Anthropic request format.
        Alias for prepare_data for standardized interface.

        Note: Image support is built into this method. Anthropic supports
        both data URIs (base64) and HTTP/HTTPS URLs for images.
        """
        return self.prepare_data(chat, **kwargs)

    def transform_response(self, raw: Dict[str, Any]) -> ModelChatResponse:
        """
        Transform Anthropic response to ModelChatResponse.
        Uses shared response_mapping utilities.
        """
        return self.process_generate(raw)

    def transform_stream_chunk(
        self,
        raw: Any,
        context: Optional[Dict] = None
    ) -> Optional[ChatCompletionModel]:
        """
        Transform Anthropic streaming event to ChatCompletionModel.

        Args:
            raw: Raw chunk string from Anthropic stream
            context: Dict with 'idx' and 'usage' for stateful transformations

        Returns:
            ChatCompletionModel or None if chunk should be skipped
        """
        context = context or {}
        model, _, _ = self.process_chunk(
            raw,
            context.get('idx'),
            context.get('usage')
        )
        return model

    def process_chunk(self, chunk: str, idx, usage):
        model, idx, usage = self.prepare_chunk(json.loads(chunk), idx, usage)
        return model, idx, usage

    def process_generate(self, claude_response: dict) -> ModelChatResponse:
        """Convert Claude API response to ModelChatResponse format"""

        # Extract tool calls if present
        tool_calls = None
        content = None

        if claude_response.get('content'):
            # Claude returns content as a list
            for item in claude_response['content']:
                if item['type'] == 'tool_use':
                    # Convert Claude tool use to OpenAI-style tool call
                    if tool_calls is None:
                        tool_calls = []

                    tool_call = build_tool_call(
                        id=item['id'],
                        name=item['name'],
                        arguments=json.dumps(item['input'])  # Convert dict to JSON string
                    )
                    tool_calls.append(tool_call)
                elif item['type'] == 'text':
                    # If there's text content, use it
                    content = item.get('text', '')

        # Map stop_reason to finish_reason using shared mapping
        finish_reason = map_finish_reason(
            claude_response.get('stop_reason'),
            ANTHROPIC_FINISH_REASON_MAP,
            default=claude_response.get('stop_reason')
        )

        # Create usage model (include cache tokens for accurate accounting)
        usage_meta = claude_response['usage']
        cache_read = usage_meta.get('cache_read_input_tokens', 0)
        cache_creation = usage_meta.get('cache_creation_input_tokens', 0)
        prompt_tokens = usage_meta['input_tokens'] + cache_read + cache_creation
        usage = UsageModel(
            prompt_tokens=prompt_tokens,
            completion_tokens=usage_meta['output_tokens'],
            total_tokens=prompt_tokens + usage_meta['output_tokens'],
            prompt_tokens_details=PromptTokensDetailsModel(cached_tokens=cache_read),
        )

        # Build standardized response
        return build_response(
            id=claude_response['id'],
            model=claude_response['model'],
            content=content,
            role=claude_response['role'],
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
            service_tier=claude_response['usage'].get('service_tier')
        )

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.base_url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout', 30))

            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.base_url,
                                        data=json_data,
                                        headers=headers,
                                        timeout=kwargs.get('timeout', 30))
            return self.process_generate(response)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        with HttpClient() as client:
            # Reset streaming block state per request
            self._active_blocks = {}
            self._finish_reason = None
            idx = None
            usage = None
            for chunk in client.stream_request("POST",
                                               self.base_url,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout', 30)):
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
            choice = ChoiceModel(delta=delta, finish_reason=self._finish_reason, index=0)
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
            # Reset streaming block state per request
            self._active_blocks = {}
            self._finish_reason = None
            idx = None
            usage = None
            async for chunk in client.post_stream(self.base_url,
                                                  data=json_data,
                                                  headers=headers,
                                                  timeout=kwargs.get('timeout', 30)):
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
            choice = ChoiceModel(delta=delta, finish_reason=self._finish_reason, index=0)
            yield ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage or UsageModel(),
            )
