import base64
import logging
import json
import inspect
from copy import deepcopy
from typing import Any, Union

from magic_llm.exception.ChatException import ChatException, RequestValidationError
from magic_llm.model import ModelChatResponse
from magic_llm.util.tokenizer import from_openai

logger = logging.getLogger(__name__)


class ModelChat:
    TOKENS_PER_MESSAGE = 3
    TOKENS_PER_NAME = 1
    ASSISTANT_PRIME_TOKENS = 3  # <|start|>assistant<|message|>

    def __init__(self, system: str = None,
                 max_input_tokens: int = None,
                 extra_args=None) -> None:
        self.messages = [{"role": "system", "content": system}] if system else []
        self.max_input_tokens = max_input_tokens
        self.extra_args = extra_args
        self._require_complete_context = False
        self._provider_payload_guard = None
        self._observer_projection = None

    def require_complete_context(self) -> None:
        """Fail on context overflow instead of dropping any messages or tool pairs."""
        self._require_complete_context = True

    @property
    def complete_context_required(self) -> bool:
        return self._require_complete_context

    def set_provider_payload_guard(self, guard) -> None:
        """Set a mandatory validator for the actual mapped provider JSON payload."""
        if guard is not None and not callable(guard):
            raise TypeError('provider payload guard must be callable')
        self._provider_payload_guard = guard

    def validate_provider_payload(self, payload: dict) -> None:
        """Called by supported adapters after final mapping, before telemetry/I/O."""
        if self._provider_payload_guard is None:
            return
        try:
            result = self._provider_payload_guard(deepcopy(payload))
            if inspect.isawaitable(result):
                close = getattr(result, 'close', None)
                if close is not None:
                    close()
                raise TypeError('provider payload guard must be synchronous')
        except RequestValidationError:
            raise
        except Exception as exc:
            raise RequestValidationError(exc) from exc

    def set_observer_projection(self, projection) -> None:
        """Configure callback-only chat projection; canonical inference stays intact."""
        if projection is not None and not callable(projection):
            raise TypeError('observer projection must be callable')
        self._observer_projection = projection

    def observer_projection(self):
        """Return observer history, projecting a detached copy when configured."""
        if self._observer_projection is None:
            return self
        cloned = ModelChat(max_input_tokens=self.max_input_tokens, extra_args=deepcopy(self.extra_args))
        cloned.messages = deepcopy(self.messages)
        result = self._observer_projection(cloned)
        if inspect.isawaitable(result):
            close = getattr(result, 'close', None)
            if close is not None:
                close()
            raise TypeError('observer projection must be synchronous')
        if not isinstance(result, ModelChat):
            raise TypeError('observer projection must return ModelChat')
        return result

    def set_system(self, system: str, index: int = 0):
        self.messages.insert(index, {"role": "system", "content": system})

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_user_message(self, content: str,
                         image: Union[str, bytes, list[Union[str, bytes]]] = None,
                         media_type: str = 'image/jpeg'):
        def process_image(i: Union[str, bytes], mt: str) -> dict:
            """Process a single image and return the appropriate format"""
            if isinstance(i, str):
                # Check if it's a URL
                if i.startswith(('http://', 'https://', 'data:')):
                    # Validate data URIs to ensure MIME type is present
                    if i.startswith('data:'):
                        # Expect "data:<mime>;base64,<payload>"
                        try:
                            header, _ = i.split(',', 1)
                        except ValueError:
                            raise ValueError("Invalid data URI. Expected 'data:<mime>;base64,<data>'")
                        if ';base64' not in header:
                            raise ValueError(
                                "Data URI for images must declare base64 encoding, e.g., data:image/png;base64,....")
                        mime = header[5:].split(';', 1)[0]
                        if not mime or '/' not in mime:
                            raise ValueError("Data URI must include a MIME type, e.g., data:image/png;base64,....")
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": i
                        }
                    }
                else:
                    # Assume it's already base64 encoded
                    # Require a valid media_type for raw base64
                    if not mt or not isinstance(mt, str) or '/' not in mt:
                        raise ValueError("Raw base64 image provided without a valid media_type. "
                                         "Pass media_type like 'image/png' or use a data URI 'data:image/png;base64,...'.")
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mt};base64,{i}"
                        }
                    }
            elif isinstance(i, bytes):
                # Convert bytes to base64
                if not mt or not isinstance(mt, str) or '/' not in mt:
                    raise ValueError("Bytes image requires a valid media_type (e.g., 'image/png').")
                base64_image = base64.b64encode(i).decode('utf-8')
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mt};base64,{base64_image}"
                    }
                }
            else:
                raise ValueError(f"Unsupported image type: {type(i)}")

        _content = None

        if content and image:
            # Start with text content
            _content = [{"type": "text", "text": content}]

            # Handle single image or list of images
            if isinstance(image, list):
                # Process each image in the list
                for img in image:
                    _content.append(process_image(img, media_type))
            else:
                # Process single image
                _content.append(process_image(image, media_type))

        elif content and not image:
            _content = content
        else:
            raise ValueError('Image cannot be alone')

        self.messages.append({
            "role": "user",
            "content": _content
        })

    def add_assistant_message(self, content: str, responses_output: list[dict] | None = None,
                              gemini_parts: list[dict] | None = None):
        self.messages.append({
            "role": "assistant",
            "content": content
        })

        if responses_output:
            self.messages[-1]['responses_output'] = responses_output
        if gemini_parts is not None:
            self.messages[-1]['gemini_parts'] = deepcopy(gemini_parts)

    def add_system_message(self, content: str):
        self.messages.append({
            "role": "system",
            "content": content
        })

    # Define format templates as class attributes for better maintainability
    FORMAT_TEMPLATES = {
        'generic': {
            'message_format': "{role}: {content}",
            'separator': "\n",
            'suffix': '\nassistant: ',
            'role_mapping': {}
        },
        'titan': {
            'message_format': "{role}: {content}",
            'separator': "\n",
            'suffix': '\nAssistant: ',
            'role_mapping': {'user': 'User'}
        },
        'claude': {
            'message_format': "{role}: {content}",
            'separator': "\n\n",
            'suffix': '\n\nAssistant: ',
            'role_mapping': {'user': 'Human', 'assistant': 'Assistant'}
        },
        'llama2': {
            'message_format': "{content}" if "{role}" == "assistant" else "[INST]{content}[/INST]",
            'separator': "\n",
            'suffix': '\n',
            'role_mapping': {},
            'special_format': True
        }
    }

    def generic_chat(self, format: str = 'generic'):
        """
        Format the chat messages according to the specified format.

        Args:
            format: The format to use ('generic', 'titan', 'claude', 'llama2')

        Returns:
            The formatted chat string

        Raises:
            ValueError: If an unsupported format is specified
        """
        messages = self.get_messages()

        # Get the format template or raise an error for unsupported formats
        if format not in self.FORMAT_TEMPLATES:
            supported_formats = ", ".join(self.FORMAT_TEMPLATES.keys())
            raise ValueError(f"Unsupported format: {format}. Supported formats: {supported_formats}")

        template = self.FORMAT_TEMPLATES[format]

        # Handle special formats like llama2 that need custom processing
        if template.get('special_format'):
            if format == 'llama2':
                return "\n".join([
                    f"{message['content']}"
                    if message['role'] in {'assistant'} else
                    f"[INST]{message['content']}[/INST]"
                    for message in messages
                ]) + template['suffix']
            # Add other special formats here as needed

        # Standard format processing
        formatted_messages = []
        for message in messages:
            # Apply role mapping if defined
            role = message['role']
            for original, replacement in template['role_mapping'].items():
                role = role.replace(original, replacement)

            # Format the message
            formatted_message = template['message_format'].format(
                role=role,
                content=message['content']
            )
            formatted_messages.append(formatted_message)

        # Join messages with the separator and add the suffix
        return template['separator'].join(formatted_messages) + template['suffix']

    def __str__(self):
        return "\n".join([f"{message['role']}: {message['content']}" for message in self.get_messages()])
        # return self.num_tokens_from_messages()

    def _estimate_image_tokens(self, image_url: str) -> int:
        """
        Estimate token count for an image based on base64 payload size.

        For data URIs: extracts base64 payload and estimates tokens.
        For HTTP URLs: uses a fixed low estimate (provider fetches separately).

        Base64 token estimation: ~4 characters per token (tiktoken approximation).
        """
        if image_url.startswith('data:'):
            try:
                _, payload = image_url.split(',', 1)
                return len(payload) // 4
            except ValueError:
                return 85
        return 85

    def _count_content_tokens(self, content) -> int:
        """
        Count tokens for message content, handling both string and multimodal (list) formats.

        Args:
            content: Either a string or a list of content parts (multimodal)

        Returns:
            Estimated token count
        """
        if isinstance(content, str):
            return len(from_openai(content))
        elif isinstance(content, list):
            tokens = 0
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        tokens += len(from_openai(part.get('text', '')))
                    elif part.get('type') == 'image_url':
                        url = part.get('image_url', {}).get('url', '')
                        tokens += self._estimate_image_tokens(url)
                    else:
                        # Native tool-result blocks and opaque provider replay.
                        tokens += len(from_openai(json.dumps(part, default=str)))
                elif isinstance(part, str):
                    tokens += len(from_openai(part))
            return tokens
        return 0

    def num_tokens_from_messages(self, messages: list[dict] = None) -> int:
        """
        Calculate the total number of tokens in messages.

        Handles both string content and multimodal content (lists with text/image parts).
        Image tokens are estimated based on base64 payload size.

        Args:
            messages: Optional list of messages. If None, uses self.messages

        Returns:
            int: Total number of tokens
        """

        num_tokens = 0
        for message in self.messages if messages is None else messages:
            num_tokens += self.TOKENS_PER_MESSAGE
            for key, value in message.items():
                if key == 'content':
                    num_tokens += self._count_content_tokens(value)
                elif isinstance(value, str):
                    num_tokens += len(from_openai(value))
                elif value is not None:
                    num_tokens += len(from_openai(json.dumps(value, default=str)))
                if key == "name":
                    num_tokens += self.TOKENS_PER_NAME

        return num_tokens + self.ASSISTANT_PRIME_TOKENS

    def get_messages(self) -> list[dict]:
        """
        Get messages while respecting token limits and preserving system messages.
        System messages are always kept, other messages are truncated if needed.

        Returns:
            List of messages that fit within token limit
        """
        if not self.messages:
            raise ChatException(
                message="No messages available to process",
                error_code='NO_MESSAGES'
            )

        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ChatException(
                message="Invalid token limit specified",
                error_code='INVALID_TOKEN_LIMIT'
            )

        if self.max_input_tokens is None:
            return self.messages

        total_tokens = self.num_tokens_from_messages()
        if total_tokens <= self.max_input_tokens:
            return self.messages

        if self._require_complete_context:
            raise RequestValidationError(ChatException(
                message="Complete request context exceeds token limit",
                error_code="COMPLETE_CONTEXT_EXCEEDS_TOKEN_LIMIT",
            ))

        system_tokens = 0
        system_messages = []

        for msg in self.messages:
            if msg['role'] == 'system':
                system_tokens += (len(from_openai(msg['content'])) +
                                  len(from_openai(msg['role'])) +
                                  self.TOKENS_PER_MESSAGE)
                system_messages.append(msg)

        if system_tokens > self.max_input_tokens:
            raise ChatException(
                message="System message exceeds token limit",
                error_code='SYSTEM_MESSAGE_EXCEEDS_TOKEN_LIMIT'
            )

        logger.info(
            f'Messages exceed token limit. Truncating from {total_tokens} to '
            f'{self.max_input_tokens} tokens (system tokens: {system_tokens})'
        )

        # Build truncated message list
        truncated_messages = []
        current_tokens = system_tokens

        for msg in reversed(self.messages):
            if msg['role'] in {'user', 'assistant'}:
                msg_tokens = (
                    self._count_content_tokens(msg['content']) +
                    len(from_openai(msg['role'])) +
                    self.TOKENS_PER_MESSAGE
                )

                if current_tokens + msg_tokens + self.ASSISTANT_PRIME_TOKENS <= self.max_input_tokens:
                    truncated_messages.append(msg)
                    current_tokens += msg_tokens
            else:
                truncated_messages.append(msg)

        final_messages = truncated_messages[::-1]
        logger.info(f'Messages truncated to {self.num_tokens_from_messages(final_messages)} tokens')
        return final_messages

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> None:
        """Append a tool result message to the conversation history.

        Args:
            tool_call_id: The provider-specific tool call identifier.
            content: The tool output content.
            is_error: Flag indicating whether this is an error result.
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "is_error": is_error,
        })

    def add_tool_call_message(
        self,
        tool_calls: list[dict[str, Any]],
        content: str | None = None,
        responses_output: list[dict] | None = None,
        gemini_parts: list[dict] | None = None,
    ) -> None:
        """Append an assistant message with tool_calls to the conversation history.

        Args:
            tool_calls: The list of tool calls from the LLM response.
            content: Optional text content from the assistant response.
        """
        self.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        if responses_output:
            self.messages[-1]['responses_output'] = responses_output
        if gemini_parts is not None:
            self.messages[-1]['gemini_parts'] = deepcopy(gemini_parts)

    def add_tool_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        """Append multiple messages to the conversation history.

        Used by adapters to inject provider-formatted tool results.

        Args:
            messages: A list of message dicts to append.
        """
        self.messages.extend(messages)

    def __add__(self, chat: 'ModelChatResponse') -> 'ModelChat':
        """
        Add a new chat message to the conversation.

        Args:
            chat: ModelChatResponse object containing the new message

        Returns:
            self: Updated ModelChat instance
        """
        self.messages.append({
            "role": chat.role,
            "content": chat.content
        })
        return self
