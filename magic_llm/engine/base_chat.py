from magic_llm.model import ModelChat


class BaseChat:
    def __init__(self,
                 model: str,
                 stream: bool = False,
                 headers: dict = None,
                 **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.stream = stream
        self.headers = headers if headers else {}

    def generate(self, chat: ModelChat, **kwargs):
        pass

    def stram_generate(self, chat: ModelChat, **kwargs):
        pass

    def embedding(self, text: list[str] | str, **kwargs):
        pass
