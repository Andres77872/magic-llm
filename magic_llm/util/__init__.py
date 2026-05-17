import asyncio
from typing import Any, Callable


def is_async_callable(fn: Callable[..., Any]) -> bool:
    """Check if a callable is async, including callable instances with async __call__.

    asyncio.iscoroutinefunction() only detects bare async def functions.
    It returns False for callable instances whose __call__ method is async.
    This helper covers both cases.

    Args:
        fn: Any callable (function, method, or callable instance).

    Returns:
        True if fn is an async callable, False otherwise.
    """
    if asyncio.iscoroutinefunction(fn):
        return True
    call_method = getattr(fn, "__call__", None)
    return call_method is not None and asyncio.iscoroutinefunction(call_method)
