from magic_llm.model import ModelChatResponse


class ModelChat:
    def __init__(self, system: str = None,
                 max_input_length: int = None,
                 extra_args=None) -> None:
        self.messages = [{"role": "system", "content": system}] if system else []
        self.max_input_length = max_input_length
        self.extra_args = extra_args

    def set_system(self, system: str, index: int = 0):
        self.messages.insert(index, {"role": "system", "content": system})

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content
        })

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
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
        return "\n".join([f"{message['role']}: {message['content']}" for message in self.messages])

    def get_messages(self):
        if self.max_input_length is None:
            return self.messages
        ctx = self.messages[0]['role'] == 'system'
        if ctx:
            c = len(self.messages[0])
        else:
            c = 0
        while c > self.max_input_length:
            if ctx:
                self.messages = [self.messages[0]] + self.messages[2:]
            else:
                self.messages = self.messages[2:]
            c += len(self.messages[-1]) + len(self.messages[-2])
        return self.messages

    def __add__(self, chat: ModelChatResponse):
        self.messages.append({
            "role": chat.role,
            "content": chat.content
        })
        return self
