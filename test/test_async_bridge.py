"""Behavioral coverage for running blocking callables from async code."""

import asyncio
import threading

import pytest

from magic_llm.util.async_bridge import run_sync_in_thread


async def test_run_sync_in_thread_keeps_event_loop_responsive():
    started = threading.Event()
    release = threading.Event()

    def blocking_call() -> int:
        started.set()
        release.wait(timeout=0.5)
        return 42

    task = asyncio.create_task(run_sync_in_thread(blocking_call))
    while not started.is_set():
        await asyncio.sleep(0)

    assert not task.done()
    release.set()
    assert await task == 42


async def test_run_sync_in_thread_propagates_callable_errors():
    def fail() -> None:
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        await run_sync_in_thread(fail)


async def test_run_sync_in_thread_enforces_timeout():
    release = threading.Event()

    def blocking_call() -> None:
        release.wait(timeout=0.5)

    try:
        with pytest.raises(asyncio.TimeoutError):
            await run_sync_in_thread(blocking_call, timeout=0.01)
    finally:
        release.set()
