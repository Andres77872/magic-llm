"""Async-safe execution helpers for blocking callables."""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import Future
from threading import Thread
from typing import Any, Callable


def submit_in_daemon_thread(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Future[Any]:
    """Start a blocking callable without attaching it to an event-loop executor."""
    future: Future[Any] = Future()
    context = contextvars.copy_context()

    def invoke() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = context.run(fn, *args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    Thread(target=invoke, name="magic-llm-worker", daemon=True).start()
    return future


async def await_thread_future(
    future: Future[Any], timeout: float | None = None
) -> Any:
    """Await a thread future without coupling it to loop-executor shutdown."""
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout

    while not future.done():
        if deadline is None:
            delay = 0.01
        else:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            delay = min(0.01, remaining)
        await asyncio.sleep(delay)

    return future.result()


async def run_sync_in_thread(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run a blocking callable off-loop and await its result."""
    future = submit_in_daemon_thread(fn, *args, **kwargs)
    return await await_thread_future(future, timeout=timeout)
