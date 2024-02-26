from magic_llm.base import MagicLlmBase

__version__ = '0.0.15'


class MagicLLM(MagicLlmBase):

    def __init__(self,
                 engine: str,
                 model: str,
                 private_key: str | None = None,
                 **kwargs):
        super().__init__(engine=engine,
                         model=model,
                         private_key=private_key,
                         **kwargs)

    def download_embedding_search_model(self):
        pass

    def download_tagger_model(self):
        pass

    def download_tags_dictionary(self):
        pass
