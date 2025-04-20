import pytest
import json

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest


def _get_chat_builder():
    chat = ModelChat()
    chat.add_user_message('hi')
    return chat


def _get_callback(msg: ModelChat,
                  content: str,
                  usage,
                  model: str,
                  meta):
    print('\n\n' + '=' * 50)
    print('==MODEL==', model)
    print('==USAGE==', usage)
    print('==GENERATED==', content)
    print('==INFO==', meta)
    print('=' * 50)


OPENAI_KEY = json.load(open('/home/andres/Documents/keys.json'))['azure']

OPENAI_KEY["callback"] = _get_callback
MODEL = '@cf/meta/llama-2-7b-chat-int8'
MODEL_FAIL = 'FAIL/@cf/meta/llama-2-7b-chat-int8'


def _get_fallback_client():
    client = MagicLLM(
        model=MODEL,
        **OPENAI_KEY
    )
    return client


@pytest.mark.asyncio
async def test_async_openai_base_stream_transcriptions_1():
    data = AudioTranscriptionsRequest(
        file=open('/home/andres/Music/speech.wav', 'rb').read(),
        language='es-MX',
    )
    k = OPENAI_KEY.copy()
    client = MagicLLM(
        **k
    )
    content = await client.llm.async_audio_transcriptions(data)
    print('content', content)
