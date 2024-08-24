from typing import Iterator, AsyncIterator, Callable, Awaitable, Optional
import abc
import functools
import asyncio
import time

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, ChatMetaModel


class BaseChat(abc.ABC):
    def __init__(
            self,
            model: str,
            headers: Optional[dict] = None,
            callback: Optional[Callable] = None,
            fallback: Optional[Callable] = None,
            **kwargs
    ):
        self.model = model
        self.headers = headers or {}
        self.callback = callback
        self.fallback = fallback
        self.kwargs = kwargs

    @staticmethod
    def _create_chat_meta_model(ttfb: float, ttf: float, usage: UsageModel) -> ChatMetaModel:
        return ChatMetaModel(
            TTFB=ttfb,
            TTF=ttf,
            TPS=usage.completion_tokens / ttf if usage else 0
        )

    @staticmethod
    def _execute_callback(
            callback: Callable,
            chat: ModelChat,
            response_content: str,
            usage: Optional[UsageModel],
            model: str,
            meta: Optional[ChatMetaModel]
    ):
        if asyncio.iscoroutinefunction(callback):
            return callback(chat, response_content, usage, model, meta)
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, callback, chat, response_content, usage, model, meta)

    @staticmethod
    def async_intercept_stream_generate(func: Callable[..., Awaitable[AsyncIterator[ChatCompletionModel]]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
            try:
                usage = None
                response_content = ''
                start_time = time.time()
                first_token_received = False
                ttfb = 0

                async for item in func(self, chat, **kwargs):
                    if not first_token_received:
                        ttfb = time.time() - start_time
                        first_token_received = True

                    if item.usage.total_tokens != 0:
                        usage = item.usage
                    response_content += item.choices[0].delta.content
                    yield item

                ttf = time.time() - start_time - ttfb
                meta = self._create_chat_meta_model(ttfb, ttf, usage)

                if self.callback:
                    await self._execute_callback(self.callback, chat, response_content, usage, self.model, meta)

            except Exception:
                if self.fallback:
                    if self.callback:
                        self.fallback.llm.callback = self.callback
                    async for i in self.fallback.llm.async_stream_generate(chat):
                        yield i
                elif self.callback:
                    await self._execute_callback(self.callback, chat, response_content, usage, self.model, None)

        return wrapper

    @staticmethod
    def sync_intercept_stream_generate(func: Callable[..., Iterator[ChatCompletionModel]]):
        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
            try:
                usage = None
                response_content = ''
                start_time = time.time()
                first_token_received = False
                ttfb = 0

                for item in func(self, chat, **kwargs):
                    if not first_token_received:
                        ttfb = time.time() - start_time
                        first_token_received = True

                    if item.usage.total_tokens != 0:
                        usage = item.usage
                    response_content += item.choices[0].delta.content
                    yield item

                ttf = time.time() - start_time - ttfb
                meta = self._create_chat_meta_model(ttfb, ttf, usage)

                if self.callback:
                    self.callback(chat, response_content, usage, self.model, meta)

            except Exception:
                if self.fallback:
                    if self.callback:
                        self.fallback.llm.callback = self.callback
                    yield from self.fallback.llm.stream_generate(chat)
                elif self.callback:
                    self.callback(chat, response_content, None, self.model, None)

        return wrapper

    @staticmethod
    def async_intercept_generate(func: Callable[..., Awaitable[ModelChatResponse]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            start_time = time.time()
            item = await func(self, chat, **kwargs)
            ttf = time.time() - start_time

            usage = UsageModel(
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                total_tokens=item.total_tokens,
            )
            meta = self._create_chat_meta_model(0, ttf, usage)

            if self.callback:
                await self._execute_callback(self.callback, chat, item.content, usage, self.model, meta)

            return item

        return wrapper

    @staticmethod
    def sync_intercept_generate(func: Callable[..., ModelChatResponse]):
        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            start_time = time.time()
            item = func(self, chat, **kwargs)
            ttf = time.time() - start_time

            usage = UsageModel(
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                total_tokens=item.total_tokens,
            )
            meta = self._create_chat_meta_model(0, ttf, usage)

            if self.callback:
                self.callback(chat, item.content, usage, self.model, meta)

            return item

        return wrapper

    @abc.abstractmethod
    def generate(self, chat: ModelChat, **kwargs):
        pass

    @abc.abstractmethod
    async def async_generate(self, chat: ModelChat, **kwargs):
        pass

    @abc.abstractmethod
    def stream_generate(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
        pass

    @abc.abstractmethod
    async def async_stream_generate(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
        pass

    def embedding(self, text: list[str] | str, **kwargs):
        pass

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        pass

    def audio_speech(SpeechRequest, **kwargs):
        pass