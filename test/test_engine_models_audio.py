import difflib
import json
import os

import pytest

from magic_llm import MagicLLM
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest, AudioSpeechRequest

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Providers with audio cap
AUDIO_PROVIDERS = [
    ("deepinfra", "openai", {"model": "openai/whisper-large-v3"}),
    ("fireworks.ai", "openai", {"model": "whisper-v3"}),
    ("groq", "openai", {"model": "whisper-large-v3"}),
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

# Locate keys file via environment variable
_KEYS_FILE = os.getenv("MAGIC_LLM_KEYS")
if not _KEYS_FILE or not os.path.exists(_KEYS_FILE):
    pytest.skip(
        "MAGIC_LLM_KEYS env var must point to a valid keys file for integration tests.",
        allow_module_level=True,
    )
with open(_KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

EXPECTED_TEXT = (
    "Dado que se trata de un único pago por el proyecto completo, debes tener en cuenta "
    "el valor a largo plazo que generará para el cliente, en lugar de solo el costo operativo o "
    "el tiempo invertido. Anteriormente, consideramos un escenario en que el cliente podía ahorrar "
    "entre 400 y 700 USD al mes en costos internos debido a la mayor precisión y eficiencia del sistema."
)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    AUDIO_PROVIDERS,
    ids=[p[0] for p in AUDIO_PROVIDERS],
)
async def test_async_audio_transcriptions(key_name, provider, kwargs):
    audio_path = os.getenv("MAGIC_LLM_AUDIO_FILE")
    if not audio_path or not os.path.exists(audio_path):
        pytest.skip("MAGIC_LLM_AUDIO_FILE env var must point to a valid .wav file.")
    keys = dict(ALL_KEYS[key_name])
    with open(audio_path, 'rb') as f:
        data = AudioTranscriptionsRequest(
            file=f.read(),
            **kwargs,
        )

    client = MagicLLM(**keys)
    resp = await client.llm.async_audio_transcriptions(data)
    received_text = resp['text'].strip().lower()
    expected_text = EXPECTED_TEXT.strip().lower()
    sim = similarity(received_text[:len(expected_text)], expected_text)
    assert sim > 0.90, f'FAIL: similitud baja ({sim:.3f})!\nEsperado: {expected_text}\nGenerado: {received_text}'

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    TTS_PROVIDERS,
    ids=[p[0] for p in TTS_PROVIDERS],
)
async def test_async_audio_speech(key_name, provider, kwargs):
    keys = dict(ALL_KEYS[key_name])

    # Build a minimal TTS request
    data = AudioSpeechRequest(
        input="Hello from MagicLLM text to speech.",
        model=kwargs.get("model", "tts-1"),
        voice=kwargs["voice"],
        response_format="mp3",
    )

    client = MagicLLM(**keys)
    audio = await client.llm.async_audio_speech(data)

    # Basic validations on returned audio bytes
    assert isinstance(audio, (bytes, bytearray))
    assert len(audio) > 1000, "Expected non-trivial audio output"