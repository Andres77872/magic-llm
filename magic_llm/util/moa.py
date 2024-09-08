from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Tuple, Any


def run(agents: List[Tuple[MagicLLM, ModelChat]], synthesizer: Callable[[List[Dict]], Any]):
    for i in agents:
        if not isinstance(i[0], MagicLLM) or not isinstance(i[1], ModelChat):
            raise TypeError(
                'Every element of the list must be a tuple of (MagicLLM, ModelChat)'
            )

    def __process__(index: int, client: MagicLLM, chat: ModelChat):
        rp = client.llm.generate(chat)
        return {
            'index': index,
            'content': rp.content
        }

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(__process__, i, m, c) for i, (m, c) in enumerate(agents)]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda x: x['index'])
    return synthesizer(results)
