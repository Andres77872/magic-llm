from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChatResponse
from magic_llm.util.tokenizer import from_openai


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

    def generic_chat(self, format: str = 'generic'):
        messages = self.get_messages()
        if format == 'generic':
            return "\n".join([
                f"{message['role']}: {message['content']}"
                for message in messages]) + '\nassistant: '
        elif format == 'titan':
            return "\n".join([
                f"{message['role'].replace('user', 'User')}: {message['content']}"
                for message in messages]) + f'\nAssistant: '
        elif format == 'claude':
            return "\n\n".join([
                f"{message['role'].replace('user', 'Human').replace('assistant', 'Assistant')}: {message['content']}"
                for message in messages]) + f'\n\nAssistant: '
        elif format == 'llama2':
            return "\n".join([
                f"{message['content']}"
                if message['role'] in {'assistant'} else
                f"[INST]{message['content']}[/INST]"
                for message in messages]) + f'\n'

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

        print(
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
        print(f'Messages truncated to {self.num_tokens_from_messages(final_messages)} tokens')
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
