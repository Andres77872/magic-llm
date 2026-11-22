import pytest
from pydantic import ValidationError

from magic_llm.engine.discovery.capabilities import (
    CompositeCapabilityInference,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
    ProviderFieldStrategy,
)
from magic_llm.model.discovery import ModelCapabilities


def test_media_capability_fields_default_false_and_are_strict():
    caps = ModelCapabilities()
    assert caps.audio_input is False
    assert caps.audio_output is False
    assert caps.image_output is False

    with pytest.raises(ValidationError):
        ModelCapabilities(image_generation=True)


def test_verified_stt_and_tts_regex_opt_ins_are_conservative():
    strategy = ModelNameRegexStrategy()
    assert strategy.infer('openai', 'whisper-1', {'id': 'whisper-1'})['audio_input'] is True
    assert strategy.infer('openai', 'gpt-4o-mini-tts', {'id': 'gpt-4o-mini-tts'})['audio_output'] is True
    assert strategy.infer('openai', 'unknown-audio-model', {'id': 'unknown-audio-model'}).get('audio_output') is None


def test_openrouter_provider_fields_distinguish_audio_input_and_output():
    strategy = ProviderFieldStrategy()
    result = strategy.infer('openrouter', 'audio-model', {
        'architecture': {
            'modality': {'input': ['text', 'audio'], 'output': ['text', 'audio']},
        },
    })
    assert result['audio_input'] is True
    assert result['audio_output'] is True
    assert 'image_output' not in result


def test_composite_keeps_image_output_false_for_media_models():
    composite = CompositeCapabilityInference([
        ProviderFieldStrategy(),
        ModelNameRegexStrategy(),
        ProviderDefaultsStrategy(),
    ])
    caps = composite.infer('openai', 'whisper-1', {'id': 'whisper-1'})
    assert caps.audio_input is True
    assert caps.audio_output is False
    assert caps.image_output is False
