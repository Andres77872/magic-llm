import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.util.http import AsyncHttpClient, HttpClient


class ProviderTogether(OpenAiBaseProvider):
    def __init__(self,
                 base_url: str = "https://api.together.xyz/v1",
                 **kwargs):
        super().__init__(
            base_url=base_url,
            **kwargs
        )
        # Together's TTS returns binary audio; relax accept header for binary content
        if self.headers.get('accept') == 'application/json':
            self.headers['accept'] = '*/*'

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        # Map AudioSpeechRequest to Together API format
        payload = {
            "model": data.model,
            "input": data.input,
            "voice": data.voice,
            "response_format": kwargs.get("response_format", data.response_format),
            "response_encoding": kwargs.get("response_encoding", "pcm_f32le"),
            "sample_rate": kwargs.get("sample_rate", 44100),
            "stream": kwargs.get("stream", False),
        }

        if 'language' in kwargs:
            payload['language'] = kwargs['language']
            
        async with AsyncHttpClient() as client:
            response = await client.post_raw_binary(
                url=self.base_url + '/audio/generations',
                json=payload,
                headers=self.headers)
            return response

    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        # Map AudioSpeechRequest to Together API format
        payload = {
            "model": data.model,
            "input": data.input,
            "voice": data.voice,
            "response_format": kwargs.get("response_format", data.response_format),
            "response_encoding": kwargs.get("response_encoding", "pcm_f32le"),
            "sample_rate": kwargs.get("sample_rate", 44100),
            "stream": kwargs.get("stream", False)
        }

        if 'language' in kwargs:
            payload['language'] = kwargs['language']
                    
        with HttpClient() as client:
            response = client.post_raw_binary(
                url=self.base_url + '/audio/generations',
                json=payload,
                headers=self.headers)
            return response
