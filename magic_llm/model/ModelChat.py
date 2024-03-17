from magic_llm.model import ModelChatResponse


class ModelChat:
    def __init__(self, system: str = None):
        self.messages = [{"role": "system", "content": system}] if system else []

    def set_system(self, system: str):
        self.messages.insert(0, {"role": "system", "content": system})

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
        if format == 'generic':
            return "\n".join([
                f"{message['role']}: {message['content']}"
                for message in self.messages]) + '\nassistant: '
        elif format == 'titan':
            return "\n".join([
                f"{message['role'].replace('user', 'User')}: {message['content']}"
                for message in self.messages]) + f'\nAssistant: '
        elif format == 'claude':
            return "\n\n".join([
                f"{message['role'].replace('user', 'Human').replace('assistant', 'Assistant')}: {message['content']}"
                for message in self.messages]) + f'\n\nAssistant: '
        elif format == 'llama2':
            return "\n".join([
                f"{message['content']}"
                if message['role'] in {'assistant'} else
                f"[INST]{message['content']}[/INST]"
                for message in self.messages]) + f'\n'

    def __str__(self):
        return "\n".join([f"{message['role']}: {message['content']}" for message in self.messages])

    def __add__(self, chat: ModelChatResponse):
        self.messages.append({
            "role": chat.role,
            "content": chat.content
        })
        return self
