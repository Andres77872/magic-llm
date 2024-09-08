from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from concurrent.futures import ThreadPoolExecutor, as_completed


def run(agents: list[tuple[MagicLLM, ModelChat]], synthesizer):
    for i in agents:
        if not isinstance(i[0], MagicLLM) or not isinstance(i[1], ModelChat):
            raise TypeError(
                'Every element of the list must be a tuple of (MagicLLM, ModelChat)'
            )
    if not isinstance(synthesizer[0], MagicLLM) or not isinstance(synthesizer[1], ModelChat):
        raise TypeError(
            'Synthesizer must be a tuple of (MagicLLM, ModelChat)'
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
    return results