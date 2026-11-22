import pytest

from magic_llm import MagicLLM
from magic_llm.engine.engine_azure import EngineAzure, AZURE_TTS_OUTPUT_FORMATS
from magic_llm.exception.ChatException import ChatException
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest


WAV_BYTES = b'RIFF\x10\x00\x00\x00WAVEfmt '


def test_public_azure_construction_accepts_speech_key_and_region():
    client = MagicLLM(engine='azure', speech_key='key', speech_region='eastus')
    assert isinstance(client.llm, EngineAzure)
    assert client.llm.speech_key == 'key'


def test_public_azure_construction_accepts_private_key_alias():
    client = MagicLLM(engine='azure', private_key='key', speech_region='eastus')
    assert client.llm.speech_key == 'key'


@pytest.mark.parametrize('kwargs, missing', [
    ({'speech_region': 'eastus'}, 'speech_key'),
    ({'speech_key': 'key'}, 'speech_region'),
])
def test_public_azure_construction_missing_config_fails(kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        MagicLLM(engine='azure', **kwargs)


def test_azure_sync_media_methods_raise_async_hint():
    engine = EngineAzure(speech_key='key', speech_region='eastus')
    speech = AudioSpeechRequest(input='hello', model='azure-speech', voice='en-US-AriaNeural')
    stt = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        language='en-US',
        filename='sample.wav',
        content_type='audio/wav',
    )

    with pytest.raises(ChatException) as tts_exc:
        engine.audio_speech(speech)
    assert 'audio_speech' in tts_exc.value.message
    assert 'async_audio_speech' in tts_exc.value.message

    with pytest.raises(ChatException) as stt_exc:
        engine.sync_audio_transcriptions(stt)
    assert 'sync_audio_transcriptions' in stt_exc.value.message
    assert 'async_audio_transcriptions' in stt_exc.value.message


def test_azure_ssml_escapes_xml_sensitive_text():
    engine = EngineAzure(speech_key='key', speech_region='eastus')
    request = AudioSpeechRequest(
        input='5 < 6 & "quoted"',
        model='azure-speech',
        voice='en-US-AriaNeural',
    )
    ssml = engine._build_ssml(request)
    assert '5 &lt; 6 &amp; &quot;quoted&quot;' in ssml
    assert '<break' not in ssml


def test_azure_tts_output_format_mapping_and_validation():
    assert EngineAzure._tts_output_format('wav') == AZURE_TTS_OUTPUT_FORMATS['wav']
    with pytest.raises(ValueError, match='Unsupported Azure TTS response_format'):
        EngineAzure._tts_output_format('aac')


def test_azure_stt_requires_language_before_request():
    engine = EngineAzure(speech_key='key', speech_region='eastus')
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        filename='sample.wav',
        content_type='audio/wav',
    )
    with pytest.raises(ValueError, match='language'):
        engine._validate_language(request.language)


def test_azure_stt_uses_accurate_wav_content_type():
    engine = EngineAzure(speech_key='key', speech_region='eastus')
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        language='en-US',
        filename='sample.wav',
        content_type='audio/wav',
    )
    assert engine._stt_headers(request)['Content-Type'] == 'audio/wav'


def test_azure_stt_rejects_non_wav_without_relabeling():
    engine = EngineAzure(speech_key='key', speech_region='eastus')
    request = AudioTranscriptionsRequest(
        file=b'ID3\x04mp3bytes',
        language='en-US',
        filename='sample.mp3',
        content_type='audio/mpeg',
    )
    with pytest.raises(ValueError, match='audio/wav'):
        engine._stt_headers(request)
