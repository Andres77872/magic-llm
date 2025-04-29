import logging
from magic_llm.exception.ChatException import ChatException
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

    def set_system(self, system: str, index: int = 0):
        self.messages.insert(index, {"role": "system", "content": system})

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_user_message(self, content: str, image: str = None, media_type: str = 'image/jpeg'):
        _content = None
        if content and image:
            _content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image}"
                    },
                },
                {
                    "type": "text", "text": content
                }
            ]

        elif content and not image:
            _content = content
        else:
            raise Exception('Image cannot be alone')
        self.messages.append({
            "role": "user",
            "content": _content
        })

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content
        })

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

    def num_tokens_from_messages(self, messages: list[dict] = None) -> int:
        """
        Calculate the total number of tokens in messages.

        Args:
            messages: Optional list of messages. If None, uses self.messages

        Returns:
            int: Total number of tokens
        """

        num_tokens = 0
        for message in messages or self.messages:
            num_tokens += self.TOKENS_PER_MESSAGE
            for key, value in message.items():
                # TODO: Add image handle for the tokenizer
                value_str = value if isinstance(value, str) else ''
                num_tokens += len(from_openai(value_str))
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
                        len(from_openai(msg['content'])) +
                        len(from_openai(msg['role'])) +
                        self.TOKENS_PER_MESSAGE  # Base tokens per message
                )

                if current_tokens + msg_tokens + self.ASSISTANT_PRIME_TOKENS <= self.max_input_tokens:
                    truncated_messages.append(msg)
                    current_tokens += msg_tokens
            else:  # system messages
                truncated_messages.append(msg)

        final_messages = truncated_messages[::-1]
        logger.info(f'Messages truncated to {self.num_tokens_from_messages(final_messages)} tokens')
        return final_messages

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
