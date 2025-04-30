import logging
from typing import Callable, Optional

from magic_llm.base import MagicLlmBase

__version__ = '0.1.13'

logger = logging.getLogger(__name__)


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
        """
        Download the embedding search model.

        This method is intended for downloading models used for embedding-based search.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_embedding_search_model method is not yet implemented")
        raise NotImplementedError("The embedding search model download functionality is not yet implemented")

    def download_tagger_model(self):
        """
        Download the tagger model.

        This method is intended for downloading models used for tagging content.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_tagger_model method is not yet implemented")
        raise NotImplementedError("The tagger model download functionality is not yet implemented")

    def download_tags_dictionary(self):
        """
        Download the tags dictionary.

        This method is intended for downloading dictionaries used for tag mapping.
        Currently not implemented.

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        logger.warning("The download_tags_dictionary method is not yet implemented")
        raise NotImplementedError("The tags dictionary download functionality is not yet implemented")
