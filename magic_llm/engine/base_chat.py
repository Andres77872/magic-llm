from magic_llm.model import ModelChat


class BaseChat:
    def __init__(self,
                 model: str,
                 headers: dict = None,
                 **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.headers = headers if headers else {}

    def generate(self, chat: ModelChat, **kwargs):
        pass

    def stream_generate(self, chat: ModelChat, **kwargs):
        pass

    def async_stream_generate(self, chat: ModelChat, **kwargs):
        pass

    def embedding(self, text: list[str] | str, **kwargs):
        pass
