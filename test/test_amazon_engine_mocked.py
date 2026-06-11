"""Tests for EngineAmazon with mocked HTTP clients."""
import base64
import binascii
import asyncio
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.engine_amazon import EngineAmazon
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.model.ModelChatResponse import ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel


def _make_mock_prepared(url="https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/invoke"):
    """Create a mock AWSPreparedRequest for patching build_sigv4_prepared_request."""
    mock = MagicMock()
    mock.url = url
    mock.body = b'{"test": true}'
    mock.headers = {
        "Authorization": "AWS4-HMAC-SHA256 ...",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Amz-Date": "20240101T000000Z",
    }
    return mock


def _build_eventstream_frame(headers_dict, payload_bytes):
    """Build a valid AWS EventStream binary frame for testing."""
    header_bytes = b""
    for name, value in headers_dict.items():
        name_encoded = name.encode("utf-8")
        header_bytes += struct.pack("B", len(name_encoded))
        header_bytes += name_encoded
        header_bytes += struct.pack("B", 7)  # string type (AWS EventStream spec)
        if isinstance(value, str):
            value = value.encode("utf-8")
        header_bytes += struct.pack(">H", len(value))
        header_bytes += value

    headers_length = len(header_bytes)
    total_length = 8 + headers_length + 4 + len(payload_bytes) + 4
    prelude = struct.pack(">II", total_length, headers_length)
    prelude_crc = binascii.crc32(prelude) & 0xFFFFFFFF
    message_body = prelude + struct.pack(">I", prelude_crc) + header_bytes + payload_bytes
    message_crc = binascii.crc32(message_body) & 0xFFFFFFFF
    return message_body + struct.pack(">I", message_crc)


class TestEngineAmazonGenerate:
    """Test EngineAmazon.generate() with mocked HTTP."""

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_generate_returns_model_chat_response(self, mock_sigv4, mock_client_class):
        """EngineAmazon.generate() returns ModelChatResponse with mocked HTTP."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_client = MagicMock()
        mock_client.post_json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        }
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        result = engine.generate(ModelChat(system="test"))

        assert isinstance(result, ModelChatResponse)
        mock_client.post_json.assert_called_once()

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_generate_uses_correct_endpoint(self, mock_sigv4, mock_client_class):
        """EngineAmazon.generate() POSTs to the correct Bedrock invoke endpoint."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_client = MagicMock()
        mock_client.post_json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        engine.generate(ModelChat(system="test"))

        call_kwargs = mock_client.post_json.call_args[1]
        assert "bedrock-runtime.us-east-1.amazonaws.com" in call_kwargs["url"]
        assert "/invoke" in call_kwargs["url"]
        assert "invoke-with-response-stream" not in call_kwargs["url"]


class TestEngineAmazonAsyncGenerate:
    """Test EngineAmazon.async_generate() with mocked HTTP."""

    @patch("magic_llm.engine.engine_amazon.AsyncHttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_prepared_request")
    def test_async_generate_returns_model_chat_response(self, mock_prepared, mock_client_class):
        """EngineAmazon.async_generate() returns ModelChatResponse with mocked HTTP."""
        mock_prepared.return_value = _make_mock_prepared()

        mock_client = AsyncMock()
        mock_client.post_json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        }
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        result = asyncio.run(engine.async_generate(ModelChat(system="test")))

        assert isinstance(result, ModelChatResponse)


class TestEngineAmazonStreamGenerate:
    """Test EngineAmazon.stream_generate() with mocked HTTP and EventStream."""

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_stream_generate_yields_chunks(self, mock_sigv4, mock_client_class):
        """EngineAmazon.stream_generate() yields ChatCompletionModel chunks."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        # Build a realistic Bedrock streaming response
        provider_event = {
            "contentBlockDelta": {"delta": {"text": "Hello"}},
            "index": 0,
        }
        bedrock_payload = {
            "bytes": base64.b64encode(json.dumps(provider_event).encode()).decode(),
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 5,
                "outputTokenCount": 3,
            },
        }
        payload_bytes = json.dumps(bedrock_payload).encode()

        frame = _build_eventstream_frame(
            headers_dict={
                ":event-type": "chunk",
                ":content-type": "application/json",
            },
            payload_bytes=payload_bytes,
        )

        mock_client = MagicMock()
        mock_client.stream_request_bytes.return_value = [frame]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        chunks = list(engine.stream_generate(
            ModelChat(system="test")
        ))

        assert len(chunks) >= 1
        assert isinstance(chunks[0], ChatCompletionModel)

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_stream_generate_uses_streaming_endpoint(self, mock_sigv4, mock_client_class):
        """EngineAmazon.stream_generate() uses invoke-with-response-stream endpoint."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_client = MagicMock()
        mock_client.stream_request_bytes.return_value = []
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        list(engine.stream_generate(ModelChat(system="test")))

        # stream_request_bytes is called with positional args: (method, url, ...)
        call_args = mock_client.stream_request_bytes.call_args
        assert "invoke-with-response-stream" in call_args[0][1]


class TestEngineAmazonAsyncStreamGenerate:
    """Test EngineAmazon.async_stream_generate() with mocked HTTP."""

    @patch("magic_llm.engine.engine_amazon.AsyncHttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_prepared_request")
    def test_async_stream_generate_yields_chunks(self, mock_prepared, mock_client_class):
        """EngineAmazon.async_stream_generate() yields ChatCompletionModel chunks."""
        stream_url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/invoke-with-response-stream"
        mock_prepared.return_value = _make_mock_prepared(url=stream_url)

        provider_event = {
            "contentBlockDelta": {"delta": {"text": "Async Hello"}},
            "index": 0,
        }
        bedrock_payload = {
            "bytes": base64.b64encode(json.dumps(provider_event).encode()).decode(),
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 10,
                "outputTokenCount": 5,
            },
        }
        payload_bytes = json.dumps(bedrock_payload).encode()

        frame = _build_eventstream_frame(
            headers_dict={
                ":event-type": "chunk",
                ":content-type": "application/json",
            },
            payload_bytes=payload_bytes,
        )

        async def mock_stream(**kwargs):
            yield frame

        mock_client = AsyncMock()
        mock_client.post_stream = mock_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        async def _collect():
            chunks = []
            async for chunk in engine.async_stream_generate(ModelChat(system="test")):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())

        assert len(chunks) >= 1
        assert isinstance(chunks[0], ChatCompletionModel)


class TestEngineAmazonAudioSpeech:
    """Test EngineAmazon.audio_speech() with mocked HTTP."""

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_audio_speech_returns_bytes(self, mock_sigv4, mock_client_class):
        """EngineAmazon.audio_speech() returns raw audio bytes."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_client = MagicMock()
        mock_client.post_raw_binary.return_value = b"RIFF....WAVE audio data"
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        from magic_llm.model.ModelAudio import AudioSpeechRequest
        data = AudioSpeechRequest(
            voice="Joanna",
            response_format="mp3",
            input="Hello world",
            model="standard",
        )
        result = engine.audio_speech(data)

        assert isinstance(result, bytes)
        assert result == b"RIFF....WAVE audio data"
        sigv4_kwargs = mock_sigv4.call_args.kwargs
        assert sigv4_kwargs["service"] == "polly"
        assert sigv4_kwargs["region"] == "us-east-1"
        body = json.loads(sigv4_kwargs["body"])
        assert body == {
            "VoiceId": "Joanna",
            "OutputFormat": "mp3",
            "Text": "Hello world",
            "Engine": "standard",
        }
        call_kwargs = mock_client.post_raw_binary.call_args.kwargs
        assert call_kwargs["url"] == "https://polly.us-east-1.amazonaws.com/v1/speech"
        assert json.loads(call_kwargs["data"]) == body
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_audio_speech_uses_polly_endpoint(self, mock_sigv4, mock_client_class):
        """EngineAmazon.audio_speech() uses Polly endpoint."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_client = MagicMock()
        mock_client.post_raw_binary.return_value = b"audio"
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        from magic_llm.model.ModelAudio import AudioSpeechRequest
        data = AudioSpeechRequest(voice="Joanna", response_format="mp3", input="Hi", model="standard")
        engine.audio_speech(data)

        call_kwargs = mock_client.post_raw_binary.call_args[1]
        assert "polly.us-east-1.amazonaws.com" in call_kwargs["url"]

    @patch("magic_llm.engine.engine_amazon.HttpClient")
    @patch("magic_llm.engine.engine_amazon.build_sigv4_headers")
    def test_audio_speech_proves_polly_body_and_signing_service(self, mock_sigv4, mock_client_class):
        """Polly proof asserts body and SigV4 metadata, not just canned bytes."""
        mock_sigv4.return_value = {"Authorization": "AWS4-HMAC-SHA256 signed"}

        mock_client = MagicMock()
        mock_client.post_raw_binary.return_value = b"audio"
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client_class.return_value = mock_client

        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )

        data = AudioSpeechRequest(
            voice="Matthew",
            response_format="ogg_vorbis",
            input="Boundary proof",
            model="neural",
        )

        assert engine.audio_speech(data) == b"audio"

        sigv4_kwargs = mock_sigv4.call_args.kwargs
        assert sigv4_kwargs["method"] == "POST"
        assert sigv4_kwargs["service"] == "polly"
        assert sigv4_kwargs["aws_access_key_id"] == "test-key"
        assert sigv4_kwargs["aws_secret_access_key"] == "test-secret"
        assert json.loads(sigv4_kwargs["body"]) == {
            "VoiceId": "Matthew",
            "OutputFormat": "ogg_vorbis",
            "Text": "Boundary proof",
            "Engine": "neural",
        }
        call_kwargs = mock_client.post_raw_binary.call_args.kwargs
        assert call_kwargs["url"] == sigv4_kwargs["url"]
        assert call_kwargs["data"] == sigv4_kwargs["body"]
        assert call_kwargs["headers"]["Authorization"].startswith("AWS4-HMAC")
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_unsupported_async_tts_fails_fast(self):
        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        with pytest.raises(ChatException) as exc_info:
            await engine.async_audio_speech(AudioSpeechRequest(voice="Joanna", response_format="mp3", input="Hi", model="standard"))
        assert exc_info.value.error_code == "UNSUPPORTED_MEDIA_OPERATION"
        assert "audio_speech for Amazon Polly sync TTS" in exc_info.value.message

    @pytest.mark.parametrize("method", ["sync_audio_transcriptions", "async_audio_transcriptions"])
    def test_unsupported_amazon_stt_modes_fail_fast(self, method):
        engine = EngineAmazon(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        request = AudioTranscriptionsRequest(
            file=b"RIFF\x10\x00\x00\x00WAVEfmt ",
            model="whisper-1",
            filename="sample.wav",
            content_type="audio/wav",
        )

        if method == "sync_audio_transcriptions":
            with pytest.raises(ChatException) as exc_info:
                engine.sync_audio_transcriptions(request)
        else:
            async def _call():
                with pytest.raises(ChatException) as exc_info:
                    await engine.async_audio_transcriptions(request)
                return exc_info.value

            exc = asyncio.run(_call())
            assert exc.error_code == "UNSUPPORTED_MEDIA_OPERATION"
            return

        assert exc_info.value.error_code == "UNSUPPORTED_MEDIA_OPERATION"
        assert "Amazon Transcribe is out of scope" in exc_info.value.message


class TestEngineAmazonBackwardCompatibility:
    """Test backward compatibility of Amazon engine constructor."""

    def test_accepts_optional_credentials(self):
        """EngineAmazon accepts optional credentials (for ambient IAM resolution)."""
        engine = EngineAmazon(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        assert engine.provider.aws_access_key_id is None
        assert engine.provider.aws_secret_access_key is None

    def test_accepts_explicit_credentials(self):
        """EngineAmazon still accepts explicit credentials."""
        engine = EngineAmazon(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        assert engine.provider.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert engine.provider.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
