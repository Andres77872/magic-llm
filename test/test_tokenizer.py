import pytest
from tokenizers import Encoding

from magic_llm.util import tokenizer


def test_sync_hf_tokenizer_returns_tokens():
    """Verify tokenizer.from_hf returns a valid Encoding with non-empty tokens."""
    res: Encoding = tokenizer.from_hf(
        'meta-llama/Llama-3.1-70B-Instruct',
        'ML in Sales: Developed and maintained the Celonis Quality Index model.',
    )
    assert isinstance(res, Encoding)
    assert len(res.ids) > 0
    assert len(res.tokens) > 0
    assert len(res.ids) == len(res.tokens)
