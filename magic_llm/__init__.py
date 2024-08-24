from magic_llm.base import MagicLlmBase
from typing import Callable, Optional

__version__ = '0.0.55'


class MagicLLM(MagicLlmBase):

    def __init__(self,
                 engine: str,
                 model: str,
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
