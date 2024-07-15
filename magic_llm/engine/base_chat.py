from typing import Iterator, AsyncIterator, Callable, Awaitable
import abc
import functools

from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel


class BaseChat(abc.ABC):
    def __init__(self,
                 model: str,
                 headers: dict = None,
                 callback: Callable = None,
                 **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.headers = headers if headers else {}
        self.callback = callback

    @staticmethod
    def async_intercept_stream_generate(func: Callable[..., Awaitable[AsyncIterator[ChatCompletionModel]]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
            usage = None
            async for item in func(self, chat, **kwargs):
                if item.usage.total_tokens != 0:
                    usage = item.usage
                yield item
            if self.callback:
                self.callback(chat, usage)

        return wrapper

    def generate(self, chat: ModelChat, **kwargs):
        pass

    def async_generate(self, chat: ModelChat, **kwargs):
        pass

    def stream_generate(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
        pass

    @abc.abstractmethod
    def async_stream_generate(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
        pass

    def embedding(self, text: list[str] | str, **kwargs):
        pass
