"""Tests for BaseChat interceptor decorators.

Covers: retry logic, fallback activation, callback execution, metrics calculation.
Uses London-style mocking — decorators are pure orchestration, not provider logic.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from magic_llm.engine.base_chat import BaseChat, Metrics, RetryConfig
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, UsageModel
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel, ChoiceModel, DeltaModel, ChatMetaModel,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_chat():
    chat = ModelChat()
    chat.add_user_message("test")
    return chat


def _make_response(content="hi", tokens=5):
    return ModelChatResponse(
        id="r1",
        object="chat.completion",
        created=1700000000,
        model="test",
        choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
        usage=UsageModel(prompt_tokens=10, completion_tokens=tokens, total_tokens=15),
    )


def _make_chunk(content="hi", finish_reason=None, usage=None):
    return ChatCompletionModel(
        id="c1",
        model="test",
        choices=[ChoiceModel(index=0, delta=DeltaModel(content=content), finish_reason=finish_reason)],
        usage=usage or UsageModel(),
    )


def _make_chunk_with_usage(content="hi", prompt=10, completion=5):
    u = UsageModel(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    return _make_chunk(content=content, usage=u)


class _ConcreteChat(BaseChat):
    """Minimal concrete subclass for testing decorators."""

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    def stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError

    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Slice 10 — sync_intercept_generate: success path
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncInterceptGenerateSuccess:
    """sync_intercept_generate — response passes through, callback fires, metrics set."""

    def test_response_passes_through(self):
        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat()
        result = wrapped(instance, _make_chat())
        assert result is resp

    def test_callback_executed_sync(self):
        callback_calls = []

        def cb(chat, content, usage, model, meta):
            callback_calls.append((content, model, meta))

        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat(callback=cb)
        wrapped(instance, _make_chat())

        assert len(callback_calls) == 1
        content, model, meta = callback_calls[0]
        assert content == "hi"
        assert meta is not None
        assert meta.TTF > 0

    def test_callback_executed_async_from_sync(self):
        """Async callbacks are executed from sync context via thread."""
        callback_calls = []

        async def async_cb(chat, content, usage, model, meta):
            callback_calls.append((content, model))

        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat(callback=async_cb)
        wrapped(instance, _make_chat())

        # Give the thread time to complete
        time.sleep(0.1)
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "hi"

    def test_metrics_set_on_usage(self):
        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat()
        wrapped(instance, _make_chat())

        assert resp.usage.ttf > 0
        assert resp.usage.ttft >= 0
        assert resp.usage.tps >= 0

    def test_no_callback_when_none(self):
        """When callback is None, nothing fires."""
        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat(callback=None)
        result = wrapped(instance, _make_chat())
        assert result is resp


# ═══════════════════════════════════════════════════════════════════════════
# Slice 11 — sync_intercept_generate: retry + exhaustion
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncInterceptGenerateRetry:
    """sync_intercept_generate — retry on failure, exhaustion raises ChatException."""

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep):
        call_count = [0]

        def flaky(s, c, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return _make_response()

        wrapped = BaseChat.sync_intercept_generate(flaky)
        instance = _ConcreteChat(retries=3)
        result = wrapped(instance, _make_chat())

        assert call_count[0] == 3
        assert result.content == "hi"
        assert mock_sleep.call_count == 2  # slept between attempts 1→2 and 2→3

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_exhaustion_raises_chat_exception(self, mock_sleep):
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: (_ for _ in ()).throw(ConnectionError("boom")))
        instance = _ConcreteChat(retries=2)

        with pytest.raises(ChatException) as exc_info:
            wrapped(instance, _make_chat())

        assert "Generation failed after 2 attempts" in str(exc_info.value.message)
        assert exc_info.value.error_code == "STREAM_GENERATION_ERROR"
        assert mock_sleep.call_count == 1  # slept once between attempt 1 and 2

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_callback_notified_on_each_failure(self, mock_sleep):
        callback_calls = []

        def cb(chat, content, usage, model, meta):
            callback_calls.append(meta.status if meta else None)

        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: (_ for _ in ()).throw(ConnectionError("boom")))
        instance = _ConcreteChat(callback=cb, retries=2)

        with pytest.raises(ChatException):
            wrapped(instance, _make_chat())

        # Both failure callbacks should have ERROR status
        error_calls = [c for c in callback_calls if c and c.startswith("ERROR")]
        assert len(error_calls) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Slice 12 — sync_intercept_generate: fallback activation
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncInterceptGenerateFallback:
    """sync_intercept_generate — fallback invoked on exhaustion."""

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_fallback_called_on_exhaustion(self, mock_sleep):
        fallback_resp = _make_response(content="fallback answer", tokens=3)
        fallback_llm = MagicMock()
        fallback_llm.generate.return_value = fallback_resp
        fallback = MagicMock()
        fallback.llm = fallback_llm

        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: (_ for _ in ()).throw(ConnectionError("boom")))
        instance = _ConcreteChat(fallback=fallback, retries=2)

        result = wrapped(instance, _make_chat())

        assert result is fallback_resp
        fallback_llm.generate.assert_called_once()

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_fallback_receives_same_args(self, mock_sleep):
        fallback_llm = MagicMock()
        fallback_llm.generate.return_value = _make_response(content="fb")
        fallback = MagicMock()
        fallback.llm = fallback_llm

        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: (_ for _ in ()).throw(ConnectionError("boom")))
        instance = _ConcreteChat(fallback=fallback, retries=1)
        chat = _make_chat()

        wrapped(instance, chat, model="test-model")

        call_args = fallback_llm.generate.call_args
        assert call_args[0][0] is chat
        assert call_args[1].get("model") == "test-model"


# ═══════════════════════════════════════════════════════════════════════════
# Slice 13 — async_intercept_generate: success, retry, fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestAsyncInterceptGenerateSuccess:
    """async_intercept_generate — response passes through, callback fires, metrics set."""

    @pytest.mark.asyncio
    async def test_response_passes_through(self):
        resp = _make_response()
        wrapped = BaseChat.async_intercept_generate(lambda s, c, **kw: asyncio.coroutine(lambda: resp)())
        # Use a proper async mock instead
        async def gen(s, c, **kw):
            return resp

        wrapped = BaseChat.async_intercept_generate(gen)
        instance = _ConcreteChat()
        result = await wrapped(instance, _make_chat())
        assert result is resp

    @pytest.mark.asyncio
    async def test_callback_executed(self):
        callback_calls = []

        async def cb(chat, content, usage, model, meta):
            callback_calls.append((content, model))

        async def gen(s, c, **kw):
            return _make_response()

        wrapped = BaseChat.async_intercept_generate(gen)
        instance = _ConcreteChat(callback=cb)
        await wrapped(instance, _make_chat())

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "hi"

    @pytest.mark.asyncio
    async def test_metrics_set_on_usage(self):
        async def gen(s, c, **kw):
            return _make_response()

        wrapped = BaseChat.async_intercept_generate(gen)
        instance = _ConcreteChat()
        result = await wrapped(instance, _make_chat())

        assert result.usage.ttf > 0
        assert result.usage.ttft >= 0


class TestAsyncInterceptGenerateRetry:
    """async_intercept_generate — retry on failure, exhaustion raises ChatException."""

    @pytest.mark.asyncio
    @patch("magic_llm.engine.base_chat.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_then_succeeds(self, mock_sleep):
        call_count = [0]

        async def flaky(s, c, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return _make_response()

        wrapped = BaseChat.async_intercept_generate(flaky)
        instance = _ConcreteChat(retries=3)
        result = await wrapped(instance, _make_chat())

        assert call_count[0] == 3
        assert result.content == "hi"

    @pytest.mark.asyncio
    @patch("magic_llm.engine.base_chat.asyncio.sleep", new_callable=AsyncMock)
    async def test_exhaustion_raises_chat_exception(self, mock_sleep):
        async def failer(s, c, **kw):
            raise ConnectionError("boom")

        wrapped = BaseChat.async_intercept_generate(failer)
        instance = _ConcreteChat(retries=2)

        with pytest.raises(ChatException) as exc_info:
            await wrapped(instance, _make_chat())

        assert "Generation failed after 2 attempts" in str(exc_info.value.message)


class TestAsyncInterceptGenerateFallback:
    """async_intercept_generate — fallback invoked on exhaustion."""

    @pytest.mark.asyncio
    @patch("magic_llm.engine.base_chat.asyncio.sleep", new_callable=AsyncMock)
    async def test_fallback_called_on_exhaustion(self, mock_sleep):
        fallback_resp = _make_response(content="fallback answer")
        fallback_llm = AsyncMock()
        fallback_llm.async_generate = AsyncMock(return_value=fallback_resp)
        fallback = MagicMock()
        fallback.llm = fallback_llm

        async def failer(s, c, **kw):
            raise ConnectionError("boom")

        wrapped = BaseChat.async_intercept_generate(failer)
        instance = _ConcreteChat(fallback=fallback, retries=2)

        result = await wrapped(instance, _make_chat())

        assert result is fallback_resp
        fallback_llm.async_generate.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Slice 14 — stream decorators: success path
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncStreamGenerateSuccess:
    """sync_intercept_stream_generate — all items yielded, TTFB set, callback fired."""

    def test_all_items_yielded(self):
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk_with_usage("c")]
        wrapped = BaseChat.sync_intercept_stream_generate(lambda s, c, **kw: iter(chunks))
        instance = _ConcreteChat()
        results = list(wrapped(instance, _make_chat()))

        # Exactly 3 items — no double-yield, metrics updated in-place on last item
        assert len(results) == 3
        assert results[0].choices[0].delta.content == "a"
        assert results[1].choices[0].delta.content == "b"

    def test_ttfb_set_on_first_token(self):
        chunks = [_make_chunk("hello"), _make_chunk_with_usage("world")]
        wrapped = BaseChat.sync_intercept_stream_generate(lambda s, c, **kw: iter(chunks))
        instance = _ConcreteChat()
        results = list(wrapped(instance, _make_chat()))

        # The last item has metrics updated in-place (no re-yield)
        final = results[-1]
        assert final.usage.ttft > 0

    def test_callback_receives_accumulated_content(self):
        callback_calls = []

        def cb(chat, content, usage, model, meta):
            callback_calls.append(content)

        chunks = [_make_chunk("Hello"), _make_chunk(" "), _make_chunk_with_usage("world")]
        wrapped = BaseChat.sync_intercept_stream_generate(lambda s, c, **kw: iter(chunks))
        instance = _ConcreteChat(callback=cb)
        list(wrapped(instance, _make_chat()))

        assert len(callback_calls) == 1
        assert callback_calls[0] == "Hello world"


class TestAsyncStreamGenerateSuccess:
    """async_intercept_stream_generate — all items yielded, TTFB set, callback fired."""

    @pytest.mark.asyncio
    async def test_all_items_yielded(self):
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk_with_usage("c")]

        async def gen(s, c, **kw):
            for ch in chunks:
                yield ch

        wrapped = BaseChat.async_intercept_stream_generate(gen)
        instance = _ConcreteChat()
        results = [ch async for ch in wrapped(instance, _make_chat())]

        assert len(results) == 3  # 3 original — no re-yield
        assert results[0].choices[0].delta.content == "a"

    @pytest.mark.asyncio
    async def test_callback_receives_accumulated_content(self):
        callback_calls = []

        async def cb(chat, content, usage, model, meta):
            callback_calls.append(content)

        chunks = [_make_chunk("Hello"), _make_chunk(" "), _make_chunk_with_usage("world")]

        async def gen(s, c, **kw):
            for ch in chunks:
                yield ch

        wrapped = BaseChat.async_intercept_stream_generate(gen)
        instance = _ConcreteChat(callback=cb)
        _ = [ch async for ch in wrapped(instance, _make_chat())]

        assert len(callback_calls) == 1
        assert callback_calls[0] == "Hello world"


# ═══════════════════════════════════════════════════════════════════════════
# Slice 15 — stream decorators: retry + exhaustion
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncStreamGenerateRetry:
    """sync_intercept_stream_generate — retry on failure, exhaustion yields error chunk + raises."""

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_exhaustion_yields_error_chunk_then_raises(self, mock_sleep):
        def failing_gen(s, c, **kw):
            yield _make_chunk("partial")
            raise ConnectionError("boom")

        wrapped = BaseChat.sync_intercept_stream_generate(failing_gen)
        instance = _ConcreteChat(retries=1)

        with pytest.raises(ChatException) as exc_info:
            list(wrapped(instance, _make_chat()))

        assert "Stream generation failed after 1 attempts" in str(exc_info.value.message)

    @patch("magic_llm.engine.base_chat.time.sleep")
    def test_callback_notified_on_stream_failure(self, mock_sleep):
        callback_calls = []

        def cb(chat, content, usage, model, meta):
            callback_calls.append(meta.status if meta else None)

        def failing_gen(s, c, **kw):
            yield _make_chunk("partial")
            raise ConnectionError("boom")

        wrapped = BaseChat.sync_intercept_stream_generate(failing_gen)
        instance = _ConcreteChat(callback=cb, retries=1)

        with pytest.raises(ChatException):
            list(wrapped(instance, _make_chat()))

        error_calls = [c for c in callback_calls if c and c.startswith("ERROR")]
        assert len(error_calls) == 1


class TestAsyncStreamGenerateRetry:
    """async_intercept_stream_generate — retry on failure, exhaustion yields error chunk + raises."""

    @pytest.mark.asyncio
    @patch("magic_llm.engine.base_chat.asyncio.sleep", new_callable=AsyncMock)
    async def test_exhaustion_yields_error_chunk_then_raises(self, mock_sleep):
        async def failing_gen(s, c, **kw):
            yield _make_chunk("partial")
            raise ConnectionError("boom")

        wrapped = BaseChat.async_intercept_stream_generate(failing_gen)
        instance = _ConcreteChat(retries=1)

        with pytest.raises(ChatException) as exc_info:
            _ = [ch async for ch in wrapped(instance, _make_chat())]

        assert "Stream generation failed after 1 attempts" in str(exc_info.value.message)


# ═══════════════════════════════════════════════════════════════════════════
# Slice 10-15 — Callback error swallowing
# ═══════════════════════════════════════════════════════════════════════════

class TestCallbackErrorSwallowing:
    """Callback errors are logged but do not propagate to the caller."""

    def test_sync_callback_error_does_not_break_response(self):
        def bad_cb(chat, content, usage, model, meta):
            raise RuntimeError("callback crash")

        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat(callback=bad_cb)

        # Should NOT raise — callback errors are swallowed
        result = wrapped(instance, _make_chat())
        assert result is resp

    @pytest.mark.asyncio
    async def test_async_callback_error_does_not_break_response(self):
        async def bad_cb(chat, content, usage, model, meta):
            raise RuntimeError("callback crash")

        async def gen(s, c, **kw):
            return _make_response()

        wrapped = BaseChat.async_intercept_generate(gen)
        instance = _ConcreteChat(callback=bad_cb)

        result = await wrapped(instance, _make_chat())
        assert result.content == "hi"


# ═══════════════════════════════════════════════════════════════════════════
# Regression: decorators work in async test environments
# ═══════════════════════════════════════════════════════════════════════════

class TestDecoratorsInAsyncEnvironment:
    """Verify sync decorators work when called from async test context.

    This is the regression test for the asyncio.run() bug that was fixed.
    Previously, calling sync_intercept_generate from an async test would
    raise RuntimeError because asyncio.run() cannot be called when an
    event loop is already running.
    """

    @pytest.mark.asyncio
    async def test_sync_generate_from_async_context(self):
        """sync_intercept_generate works when called from async code."""
        resp = _make_response()
        wrapped = BaseChat.sync_intercept_generate(lambda s, c, **kw: resp)
        instance = _ConcreteChat()

        # This used to raise: RuntimeError: This event loop is already running
        result = wrapped(instance, _make_chat())
        assert result is resp

    @pytest.mark.asyncio
    async def test_sync_stream_from_async_context(self):
        """sync_intercept_stream_generate works when called from async code."""
        chunks = [_make_chunk("a"), _make_chunk_with_usage("b")]
        wrapped = BaseChat.sync_intercept_stream_generate(lambda s, c, **kw: iter(chunks))
        instance = _ConcreteChat()

        # This used to raise: RuntimeError: This event loop is already running
        results = list(wrapped(instance, _make_chat()))
        assert len(results) >= 2
