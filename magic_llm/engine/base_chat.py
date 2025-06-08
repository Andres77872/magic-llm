import abc
import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, AsyncIterator, Callable, Awaitable, Optional, Union, List, Any

from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
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


def is_running_in_jupyter():
    try:
        from IPython import get_ipython
        if get_ipython():
            return True
        else:
            return False
    except ImportError:
        return False


class BaseChat(abc.ABC):
    def __init__(
            self,
            model: str | None = None,
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
            model: str,
            meta: Optional[ChatMetaModel]
    ) -> None:
        """Execute callback with proper async/sync handling."""
        if not self.callback:
            return

        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(chat, response_content, usage, model, meta)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.callback,
                    chat,
                    response_content,
                    usage,
                    model,
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
            model = self.model or kwargs.get('model')
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
                        await self._execute_callback(chat, response_content, usage, model, meta)
                    break

                except Exception as e:
                    er = f"Stream generation attempt {attempt + 1} failed: {e}"
                    logger.error(er)
                    await self._execute_callback(chat,
                                                 response_content,
                                                 usage,
                                                 model,
                                                 ChatMetaModel(
                                                     TTFB=time.time() - metrics.start_time,
                                                     TTF=0,
                                                     TPS=0,
                                                     status='ERROR: ' + er))

                    if attempt == self.retry_config.attempts - 1:
                        fallback = self._handle_fallback(is_async=True)
                        if fallback:
                            async for i in fallback(chat):
                                yield i
                        else:
                            yield ChatCompletionModel(**{
                                'model': self.model,
                                'id': 'id',
                                'choices': [
                                    {
                                        'delta': {
                                            'content': '',
                                            'role': None
                                        },
                                        'finish_reason': f'error: {er}',
                                        'index': 0
                                    }
                                ]
                            })
                            raise ChatException(
                                message=f"Stream generation failed after {attempt + 1} attempts: {er}",
                                error_code='STREAM_GENERATION_ERROR',
                            )
                    else:
                        await asyncio.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def sync_intercept_stream_generate(func: Callable[..., Iterator[ChatCompletionModel]]):
        if is_running_in_jupyter():
            import nest_asyncio
            nest_asyncio.apply()

        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> Iterator[ChatCompletionModel]:
            model = self.model or kwargs.get('model')
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
                        asyncio.run(self._execute_callback(chat, response_content, usage, model, meta))
                    break

                except Exception as e:
                    er = f"Sync stream generation attempt {attempt + 1} failed: {e}"
                    logger.error(er)
                    asyncio.run(self._execute_callback(chat,
                                                       response_content,
                                                       usage,
                                                       model,
                                                       ChatMetaModel(
                                                           TTFB=time.time() - metrics.start_time,
                                                           TTF=0,
                                                           TPS=0,
                                                           status='ERROR: ' + er)))

                    if attempt == self.retry_config.attempts - 1:
                        fallback = self._handle_fallback(is_async=False)
                        if fallback:
                            yield from fallback(chat)
                        else:
                            raise ChatException(
                                message=f"Stream generation failed after {attempt + 1} attempts: {er}",
                                error_code='STREAM_GENERATION_ERROR',
                            )
                    else:
                        time.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def async_intercept_generate(func: Callable[..., Awaitable[ModelChatResponse]]):
        @functools.wraps(func)
        async def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            model = self.model or kwargs.get('model')
            for attempt in range(self.retry_config.attempts):
                start_time = time.time()
                usage = None
                try:
                    response = await func(self, chat, **kwargs)
                    ttf = time.time() - start_time

                    usage = response.usage
                    meta = self._create_chat_meta_model(0, ttf, usage)
                    usage.tps = meta.TPS
                    usage.ttf = meta.TTF
                    usage.ttft = meta.TTFB

                    await self._execute_callback(chat, response.content, usage, model, meta)
                    return response

                except Exception as e:
                    er = f"Async generation attempt {attempt + 1} failed: {e}"
                    logger.error(er)
                    await self._execute_callback(chat,
                                                 None,
                                                 usage,
                                                 model,
                                                 ChatMetaModel(
                                                     TTFB=time.time() - start_time,
                                                     TTF=0,
                                                     TPS=0,
                                                     status='ERROR: ' + er))
                    if attempt == self.retry_config.attempts - 1:
                        if self.fallback:
                            return await self.fallback.llm.async_generate(chat)
                        else:
                            raise ChatException(
                                message=f"Generation failed after {attempt + 1} attempts: {er}",
                                error_code='STREAM_GENERATION_ERROR',
                            )
                    await asyncio.sleep(self.retry_config.delay)

        return wrapper

    @staticmethod
    def sync_intercept_generate(func: Callable[..., ModelChatResponse]):
        if is_running_in_jupyter():
            import nest_asyncio
            nest_asyncio.apply()

        @functools.wraps(func)
        def wrapper(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
            model = self.model or kwargs.get('model')
            for attempt in range(self.retry_config.attempts):
                usage = None
                start_time = time.time()
                try:
                    response = func(self, chat, **kwargs)
                    ttf = time.time() - start_time

                    usage = response.usage
                    meta = self._create_chat_meta_model(0, ttf, usage)
                    usage.tps = meta.TPS
                    usage.ttf = meta.TTF
                    usage.ttft = meta.TTFB
                    asyncio.run(self._execute_callback(chat, response.content, usage, model, meta))
                    return response

                except Exception as e:
                    er = f"Sync generation attempt {attempt + 1} failed: {e}"
                    logger.error(er)
                    asyncio.run(self._execute_callback(chat,
                                                       None,
                                                       usage,
                                                       model,
                                                       ChatMetaModel(
                                                           TTFB=time.time() - start_time,
                                                           TTF=0,
                                                           TPS=0,
                                                           status='ERROR: ' + er)))

                    if attempt == self.retry_config.attempts - 1:
                        if self.fallback:
                            return self.fallback.llm.generate(chat)
                        else:
                            raise ChatException(
                                message=f"Generation failed after {attempt + 1} attempts: {er}",
                                error_code='STREAM_GENERATION_ERROR',
                            )
                    time.sleep(self.retry_config.delay)

        return wrapper

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

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

    async def async_embedding(self, text: Union[List[str], str], **kwargs) -> Any:
        """Generate embeddings for the given text asynchronously."""
        pass

    async def async_audio_speech(self, speech_request: AudioSpeechRequest, **kwargs) -> Any:
        """Generate audio speech asynchronously."""
        pass

    def audio_speech(self, speech_request: AudioSpeechRequest, **kwargs) -> Any:
        """Generate audio speech synchronously."""
        pass

    async def async_audio_transcriptions(self, speech_request: AudioTranscriptionsRequest, **kwargs) -> Any:
        """Generate audio transcriptions asynchronously."""
        pass

    def sync_audio_transcriptions(self, speech_request: AudioTranscriptionsRequest, **kwargs) -> Any:
        """Generate audio transcriptions synchronously."""
        pass
