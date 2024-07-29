from typing import Iterator, AsyncIterator, Callable, Awaitable
import abc
import functools

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class BaseChat(abc.ABC):
    def __init__(self,
                 model: str,
                 headers: dict = None,
                 callback: Callable = None,
                 fallback: Callable = None,
                 **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.headers = headers if headers else {}
        self.callback = callback
        self.fallback = fallback

    @staticmethod
    def async_intercept_stream_generate(func: Callable[..., Awaitable[AsyncIterator[ChatCompletionModel]]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
            try:
                usage = None
                async for item in func(self, chat, **kwargs):
                    if item.usage.total_tokens != 0:
                        usage = item.usage
                    yield item
                if self.callback:
                    self.callback(chat, usage)
            except:
                if self.fallback:
                    if self.callback:
                        self.fallback.llm.callback = self.callback
                    async for i in self.fallback.llm.async_stream_generate(chat):
                        yield i
                else:
                    if self.callback:
                        self.callback(chat, None)

        return wrapper

    @staticmethod
    def sync_intercept_stream_generate(func: Callable[..., Awaitable[AsyncIterator[ChatCompletionModel]]]):
        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
            try:
                usage = None
                for item in func(self, chat, **kwargs):
                    if item.usage.total_tokens != 0:
                        usage = item.usage
                    yield item
                if self.callback:
                    self.callback(chat, usage)
            except:
                if self.fallback:
                    if self.callback:
                        self.fallback.llm.callback = self.callback
                    for i in self.fallback.llm.stream_generate(chat):
                        yield i
                else:
                    self.callback(chat, None)

        return wrapper

    @staticmethod
    def async_intercept_generate(func: Callable[..., Awaitable[ModelChatResponse]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            item = await func(self, chat, **kwargs)
            usage = UsageModel(**{
                'prompt_tokens': item.prompt_tokens,
                'completion_tokens': item.completion_tokens,
                'total_tokens': item.total_tokens,
            })
            if self.callback:
                self.callback(chat, usage)
            return item

        return wrapper

    def generate(self, chat: ModelChat, **kwargs):
        pass

    @abc.abstractmethod
    def async_generate(self, chat: ModelChat, **kwargs):
        pass

    @abc.abstractmethod
    def stream_generate(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
        pass

    @abc.abstractmethod
    def async_stream_generate(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
        pass

    def embedding(self, text: list[str] | str, **kwargs):
        pass

    def async_audio_speech(self, data: AudioSpeechRequest,
                           **kwargs):
        pass

    def audio_speech(self, data: AudioSpeechRequest,
                     **kwargs):
        pass
