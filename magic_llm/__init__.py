from typing import Callable, Optional

from magic_llm.base import MagicLlmBase

__version__ = '0.1.7'


class MagicLLM(MagicLlmBase):

    def __init__(self,
                 engine: str,
                 model: str | None = None,
                 private_key: str | None = None,
                 callback: Optional[Callable] = None,
                 **kwargs):
        super().__init__(engine=engine,
                         model=model,
                         private_key=private_key,
                         callback=callback,
                         **kwargs)

    def download_embedding_search_model(self):
        pass

    def download_tagger_model(self):
        pass

    def download_tags_dictionary(self):
        pass
