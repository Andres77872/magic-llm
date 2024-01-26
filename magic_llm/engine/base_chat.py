from magic_llm.model import ModelChat


class BaseChat:
    def __init__(self):
        pass

    def generate(self, chat: ModelChat, **kwargs):
        pass

    def stram_generate(self, chat: ModelChat, **kwargs):
        pass
