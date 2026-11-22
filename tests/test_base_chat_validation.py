"""Tests for BaseChat validation methods (magic_llm.engine.base_chat)."""

import pytest

from magic_llm.engine.base_chat import BaseChat
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import ModelChatResponse, Choice, Message
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel


class _ConcreteBaseChat(BaseChat):
    """Minimal concrete subclass to test BaseChat validation methods."""

    def generate(self, chat, **kwargs):
        raise NotImplementedError

    async def async_generate(self, chat, **kwargs):
        raise NotImplementedError

    def stream_generate(self, chat, **kwargs):
        raise NotImplementedError

    async def async_stream_generate(self, chat, **kwargs):
        raise NotImplementedError


@pytest.fixture
def chat_engine():
    return _ConcreteBaseChat(model="test-model")


class TestValidateInput:
    """validate_input: must reject empty chats, pass valid ones through."""

    def test_empty_chat_raises(self, chat_engine):
        empty_chat = ModelChat()
        with pytest.raises(ChatException) as exc_info:
            chat_engine.validate_input(empty_chat)
        assert exc_info.value.error_code == "EMPTY_CHAT"

    def test_valid_chat_passes_through(self, chat_engine):
        chat = ModelChat()
        chat.add_user_message("Hello")
        result = chat_engine.validate_input(chat)
        assert result is chat

    def test_chat_with_system_message_passes(self, chat_engine):
        chat = ModelChat(system="You are helpful")
        result = chat_engine.validate_input(chat)
        assert result is chat


class TestValidateOutput:
    """validate_output: must reject responses with no choices, pass valid ones."""

    def test_empty_choices_raises(self, chat_engine):
        response = ModelChatResponse(
            id="test", object="chat.completion", created=0.0,
            model="test", choices=[],
        )
        with pytest.raises(ChatException) as exc_info:
            chat_engine.validate_output(response)
        assert exc_info.value.error_code == "EMPTY_RESPONSE"

    def test_valid_response_passes_through(self, chat_engine):
        response = ModelChatResponse(
            id="test", object="chat.completion", created=0.0,
            model="test",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
        )
        result = chat_engine.validate_output(response)
        assert result is response


class TestValidateStreamChunk:
    """validate_stream_chunk: must reject chunks with no choices, pass valid ones."""

    def test_empty_choices_raises(self, chat_engine):
        chunk = ChatCompletionModel(
            id="chunk-1", model="test", choices=[],
        )
        with pytest.raises(ChatException) as exc_info:
            chat_engine.validate_stream_chunk(chunk)
        assert exc_info.value.error_code == "EMPTY_CHUNK"

    def test_valid_chunk_passes_through(self, chat_engine):
        chunk = ChatCompletionModel(
            id="chunk-1", model="test",
            choices=[ChoiceModel(index=0, delta=DeltaModel(content="hi"))],
        )
        result = chat_engine.validate_stream_chunk(chunk)
        assert result is chunk
