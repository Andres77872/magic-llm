import difflib
import asyncio
import os

import pytest

from magic_llm import MagicLLM
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest, AudioSpeechRequest

from conftest import get_provider_key

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Providers with verified async STT support in the canonical media matrix.
AUDIO_PROVIDERS = [
    ("fireworks.ai", "openai", {"model": "whisper-v3"}),
    ("azure", "azure", {"language": "es-MX"}),
    ("openai", "openai", {"model": "whisper-1"}),
]

# Providers with text-to-speech cap (async)
TTS_PROVIDERS = [
    ("openai", "openai", {"model": "gpt-4o-mini-tts", "voice": "alloy"}),
    ("azure", "azure", {"voice": "en-US-AriaNeural"}),
    ("together.ai", "openai", {"model": "cartesia/sonic-2", "voice": "spanish narrator man", "base_url": "https://api.together.xyz/v1"}),
    ("google", "google", {"model": "gemini-2.5-flash-preview-tts", "voice": "Kore"}),
]

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

EXPECTED_TEXT = (
    "Dado que se trata de un único pago por el proyecto completo, debes tener en cuenta "
    "el valor a largo plazo que generará para el cliente, en lugar de solo el costo operativo o "
    "el tiempo invertido. Anteriormente, consideramos un escenario en que el cliente podía ahorrar "
    "entre 400 y 700 USD al mes en costos internos debido a la mayor precisión y eficiencia del sistema."
)

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    AUDIO_PROVIDERS,
    ids=[p[0] for p in AUDIO_PROVIDERS],
)
def test_async_audio_transcriptions(provider_keys, sample_audio_path, key_name, provider, kwargs):
    audio_path = sample_audio_path
    keys = get_provider_key(provider_keys, provider, key_name)
    with open(audio_path, 'rb') as f:
        data = AudioTranscriptionsRequest(
            file=f.read(),
            filename=os.path.basename(audio_path),
            content_type="audio/wav" if audio_path.lower().endswith(".wav") else None,
            **kwargs,
        )

    client = MagicLLM(**keys)
    resp = asyncio.run(client.llm.async_audio_transcriptions(data))
    received_text = resp['text'].strip().lower()
    expected_text = EXPECTED_TEXT.strip().lower()
    sim = similarity(received_text[:len(expected_text)], expected_text)
    assert sim > 0.90, f'FAIL: similitud baja ({sim:.3f})!\nEsperado: {expected_text}\nGenerado: {received_text}'

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    TTS_PROVIDERS,
    ids=[p[0] for p in TTS_PROVIDERS],
)
def test_async_audio_speech(provider_keys, key_name, provider, kwargs):
    keys = get_provider_key(provider_keys, provider, key_name)

    # Build a minimal TTS request
    data = AudioSpeechRequest(
        input="Hello from MagicLLM text to speech.",
        model=kwargs.get("model", "tts-1"),
        voice=kwargs["voice"],
        response_format="mp3",
    )

    client = MagicLLM(**keys)
    audio = asyncio.run(client.llm.async_audio_speech(data))

    # Basic validations on returned audio bytes
    assert isinstance(audio, (bytes, bytearray))
    assert len(audio) > 1000, "Expected non-trivial audio output"
