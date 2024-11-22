from typing import Iterator, AsyncIterator, Callable, Awaitable, Optional, Union, List, Any
import abc
import functools
import asyncio
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, ChatMetaModel

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    attempts: int
    delay: float = 1.0


@dataclass
class Metrics:
    start_time: float = 0.0
    ttfb: float = 0.0
    generation_time: float = 0.0

    def calculate_ttf(self) -> float:
        return time.time() - self.start_time - self.ttfb


class ChatException(Exception):
    """Base exception for chat operations."""
    pass


class BaseChat(abc.ABC):
    def __init__(
            self,
            model: str,
            headers: Optional[dict] = None,
            callback: Optional[Callable] = None,
            fallback: Optional[Callable] = None,
            retries: int = 3,
            executor: Optional[ThreadPoolExecutor] = None,
            **kwargs
    ):
        self.model = model
        self.headers = headers or {}
        self.callback = callback
        self.fallback = fallback
        self.retry_config = RetryConfig(retries)
        self.executor = executor or ThreadPoolExecutor()
        self.kwargs = kwargs

    @staticmethod
    def _create_chat_meta_model(ttfb: float, ttf: float, usage: Optional[UsageModel]) -> ChatMetaModel:
        """Create a ChatMetaModel with calculated metrics."""
        try:
            tps = usage.completion_tokens / ttf if usage and ttf > 0 else 0
            return ChatMetaModel(TTFB=ttfb, TTF=ttf, TPS=tps)
        except Exception as e:
            logger.error(f"Error creating chat meta model: {e}")
            return ChatMetaModel(TTFB=ttfb, TTF=ttf, TPS=0)

    async def _execute_callback(
            self,
            chat: ModelChat,
            response_content: str,
            usage: Optional[UsageModel],
            meta: Optional[ChatMetaModel]
    ) -> None:
        """Execute callback with proper async/sync handling."""
        if not self.callback:
            return

        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(chat, response_content, usage, self.model, meta)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.callback,
                    chat,
                    response_content,
                    usage,
                    self.model,
                    meta
                )
        except Exception as e:
            logger.error(f"Callback execution failed: {e}")

    def _update_metrics(self, item: ChatCompletionModel, metrics: Metrics, usage: Optional[UsageModel]) -> None:
        """Update metrics for a chat completion item."""
        try:
            generation_time = time.time() - metrics.generation_time
            item.usage.ttft = metrics.ttfb
            item.usage.ttf = metrics.calculate_ttf()
            if generation_time > 0 and usage:
                item.usage.tps = usage.completion_tokens / generation_time
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _handle_fallback(self, is_async: bool = False) -> Optional[Callable]:
        """Configure and return appropriate fallback handler."""
        if not self.fallback:
            return None

        if self.callback:
            self.fallback.llm.callback = self.callback

        return (
            self.fallback.llm.async_stream_generate if is_async
            else self.fallback.llm.stream_generate
        )

    @staticmethod
    def async_intercept_stream_generate(func: Callable[..., Awaitable[AsyncIterator[ChatCompletionModel]]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
            usage = None
            response_content = ''
            metrics = Metrics()

            for attempt in range(self.retry_config.attempts):
                try:
                    metrics.start_time = time.time()
                    first_token_received = False
                    current_item = None

                    async for item in func(self, chat, **kwargs):
                        current_item = item
                        if not first_token_received:
                            metrics.ttfb = time.time() - metrics.start_time
                            metrics.generation_time = time.time()
                            first_token_received = True

                        if item.usage.total_tokens:
                            usage = item.usage
                        if content := item.choices[0].delta.content:
                            response_content += content
                        yield item

                    if first_token_received and current_item:
                        self._update_metrics(current_item, metrics, usage)
                        yield current_item

                        meta = self._create_chat_meta_model(
                            metrics.ttfb,
                            metrics.calculate_ttf(),
                            usage
                        )
                        await self._execute_callback(chat, response_content, usage, meta)
                    break

                except Exception as e:
                    logger.error(f"Stream generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_config.attempts - 1:
                        fallback = self._handle_fallback(is_async=True)
                        if fallback:
                            async for i in fallback(chat):
                                yield i
                        else:
                            await self._execute_callback(chat, response_content, usage, None)
                    else:
                        await asyncio.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def sync_intercept_stream_generate(func: Callable[..., Iterator[ChatCompletionModel]]):
        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
            usage = None
            response_content = ''
            metrics = Metrics()

            for attempt in range(self.retry_config.attempts):
                try:
                    metrics.start_time = time.time()
                    first_token_received = False
                    current_item = None

                    for item in func(self, chat, **kwargs):
                        current_item = item
                        if not first_token_received:
                            metrics.ttfb = time.time() - metrics.start_time
                            metrics.generation_time = time.time()
                            first_token_received = True

                        if item.usage.total_tokens:
                            usage = item.usage
                        if content := item.choices[0].delta.content:
                            response_content += content
                        yield item

                    if first_token_received and current_item:
                        self._update_metrics(current_item, metrics, usage)
                        yield current_item

                        meta = self._create_chat_meta_model(
                            metrics.ttfb,
                            metrics.calculate_ttf(),
                            usage
                        )
                        asyncio.run(self._execute_callback(chat, response_content, usage, meta))
                    break

                except Exception as e:
                    logger.error(f"Sync stream generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_config.attempts - 1:
                        fallback = self._handle_fallback(is_async=False)
                        if fallback:
                            yield from fallback(chat)
                        else:
                            asyncio.run(self._execute_callback(chat, response_content, usage, None))
                    else:
                        time.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def async_intercept_generate(func: Callable[..., Awaitable[ModelChatResponse]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            for attempt in range(self.retry_config.attempts):
                try:
                    start_time = time.time()
                    response = await func(self, chat, **kwargs)
                    ttf = time.time() - start_time

                    usage = UsageModel(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                    )
                    meta = self._create_chat_meta_model(0, ttf, usage)
                    await self._execute_callback(chat, response.content, usage, meta)
                    return response

                except Exception as e:
                    logger.error(f"Async generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_config.attempts - 1:
                        if self.fallback:
                            return await self.fallback.llm.async_generate(chat)
                        await self._execute_callback(chat, "", None, None)
                        raise ChatException("All retry attempts failed") from e
                    await asyncio.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def sync_intercept_generate(func: Callable[..., ModelChatResponse]):
        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            for attempt in range(self.retry_config.attempts):
                try:
                    start_time = time.time()
                    response = func(self, chat, **kwargs)
                    ttf = time.time() - start_time

                    usage = UsageModel(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                    )
                    meta = self._create_chat_meta_model(0, ttf, usage)
                    asyncio.run(self._execute_callback(chat, response.content, usage, meta))
                    return response

                except Exception as e:
                    logger.error(f"Sync generation attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_config.attempts - 1:
                        if self.fallback:
                            return self.fallback.llm.generate(chat)
                        asyncio.run(self._execute_callback(chat, "", None, None))
                        raise ChatException("All retry attempts failed") from e
                    time.sleep(self.retry_config.delay)

        return wrapper

    @abc.abstractmethod
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        """Generate a chat response synchronously."""
        pass

    @abc.abstractmethod
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        """Generate a chat response asynchronously."""
        pass

    @abc.abstractmethod
    def stream_generate(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
        """Stream generate a chat response synchronously."""
        pass

    @abc.abstractmethod
    async def async_stream_generate(self, chat: ModelChat, **kwargs) -> AsyncIterator[ChatCompletionModel]:
        """Stream generate a chat response asynchronously."""
        pass

    def embedding(self, text: Union[List[str], str], **kwargs) -> Any:
        """Generate embeddings for the given text."""
        pass

    async def async_audio_speech(**kwargs) -> Any:
        """Generate audio speech asynchronously."""
        pass

    def audio_speech(self, speech_request: Any, **kwargs) -> Any:
        """Generate audio speech synchronously."""
        pass

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)