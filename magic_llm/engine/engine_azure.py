from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.util.http import AsyncHttpClient


class EngineAzure(BaseChat):
    engine = 'azure'
    def __init__(self,
                 speech_key: str,
                 speech_region: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.base_transcription_url = f"https://{speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language="
        self.speech_key = speech_key

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        # Construct SSML (Speech Synthesis Markup Language) payload
        lang = '-'.join(data.voice.split('-')[:2])
        ssml_template = f"""
        <speak version='1.0' xml:lang='{lang}'>
            <voice xml:lang='{lang}' name='{data.voice}'>
                {data.input}
            </voice>
        </speak>
        """

        # Headers required by the Microsoft Speech API
        headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            "User-Agent": "magic-audio https://arz.ai",
        }
        async with AsyncHttpClient() as client:
            response = await client.post_raw_binary(url=self.base_url,
                                                    headers=headers,
                                                    data=ssml_template.strip())
            return response

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        # TODO FIX ERROR FOR NON WAV FILES
        # Headers as required by the Microsoft Speech API
        headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "audio/wav",
            "User-Agent": "magic-transcriptions https://arz.ai",
        }

        async with AsyncHttpClient() as client:
            response = await client.post_json(
                url=self.base_transcription_url + data.language,
                headers=headers,
                data=data.file)

            return response
