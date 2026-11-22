import pytest
import tiktoken
from tokenizers import Encoding

from magic_llm.util import tokenizer


def test_openai_encoding_returns_known_token_ids_without_downloading(monkeypatch):
    # Pinned byte vocabulary with one merge; exercise real BPE offline.
    ranks = {bytes([i]): i for i in range(256)}
    ranks[b'hi'] = 256
    encoding = tiktoken.Encoding(name='tiny-test', pat_str=r'(?s).+',
                                mergeable_ranks=ranks, special_tokens={})
    monkeypatch.setattr(tiktoken, 'encoding_for_model', {'gpt-4o': encoding}.__getitem__)
    assert tokenizer.from_openai('hi!') == [256, 33]


@pytest.mark.provider_functional
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
