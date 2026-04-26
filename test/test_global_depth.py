"""Unit tests for global depth ContextVar helpers.

Tests cover:
- get_global_depth() returns 0 by default
- increment_global_depth() returns new value
- decrement_global_depth() returns new value (min 0)
- reset_global_depth() sets to 0
- ContextVar isolation across async tasks
- MAX_GLOBAL_DEPTH enforcement
"""
import asyncio

import pytest

from magic_llm.agent._loop_shared import (
    GLOBAL_DEPTH,
    get_global_depth,
    increment_global_depth,
    decrement_global_depth,
    reset_global_depth,
)
from magic_llm.agent.config import MAX_GLOBAL_DEPTH


class TestGlobalDepthHelpers:
    """Global depth ContextVar helper functions."""

    def test_get_global_depth_returns_zero_by_default(self):
        """get_global_depth() returns 0 by default."""
        reset_global_depth()
        assert get_global_depth() == 0

    def test_increment_global_depth_returns_new_value(self):
        """increment_global_depth() returns new value (1, 2, 3...)."""
        reset_global_depth()
        
        result1 = increment_global_depth()
        assert result1 == 1
        assert get_global_depth() == 1
        
        result2 = increment_global_depth()
        assert result2 == 2
        assert get_global_depth() == 2
        
        result3 = increment_global_depth()
        assert result3 == 3
        assert get_global_depth() == 3

    def test_decrement_global_depth_returns_new_value(self):
        """decrement_global_depth() returns new value (min 0)."""
        reset_global_depth()
        
        increment_global_depth()
        increment_global_depth()
        assert get_global_depth() == 2
        
        result1 = decrement_global_depth()
        assert result1 == 1
        assert get_global_depth() == 1
        
        result2 = decrement_global_depth()
        assert result2 == 0
        assert get_global_depth() == 0
        
        # Decrement below 0 should stay at 0
        result3 = decrement_global_depth()
        assert result3 == 0
        assert get_global_depth() == 0

    def test_reset_global_depth_sets_to_zero(self):
        """reset_global_depth() sets to 0."""
        increment_global_depth()
        increment_global_depth()
        assert get_global_depth() == 2
        
        reset_global_depth()
        assert get_global_depth() == 0

    @pytest.mark.asyncio
    async def test_contextvar_isolation_across_async_tasks(self):
        """ContextVar isolation across async tasks."""
        reset_global_depth()
        
        async def task1():
            increment_global_depth()
            increment_global_depth()
            await asyncio.sleep(0.01)
            return get_global_depth()
        
        async def task2():
            # Should see initial 0, not task1's increments
            initial = get_global_depth()
            increment_global_depth()
            await asyncio.sleep(0.01)
            return initial, get_global_depth()
        
        # Run concurrently
        result1, (initial2, result2) = await asyncio.gather(task1(), task2())
        
        # Each task has isolated context
        assert result1 == 2
        assert initial2 == 0
        assert result2 == 1

    def test_max_global_depth_constant(self):
        """MAX_GLOBAL_DEPTH is set correctly."""
        assert MAX_GLOBAL_DEPTH == 10

    def test_global_depth_can_reach_max(self):
        """Global depth can reach MAX_GLOBAL_DEPTH."""
        reset_global_depth()
        
        for i in range(MAX_GLOBAL_DEPTH):
            increment_global_depth()
        
        assert get_global_depth() == MAX_GLOBAL_DEPTH

    def test_global_depth_exceeds_max(self):
        """Global depth can exceed MAX_GLOBAL_DEPTH (enforcement is in TaskExecutor)."""
        reset_global_depth()
        
        for i in range(MAX_GLOBAL_DEPTH + 1):
            increment_global_depth()
        
        # ContextVar allows exceeding, but TaskExecutor enforces the limit
        assert get_global_depth() == MAX_GLOBAL_DEPTH + 1
        
        # Reset for other tests
        reset_global_depth()