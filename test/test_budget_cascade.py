"""Unit tests for _compute_child_budget() helper.

Tests cover:
- cascade disabled + nested_budget provided → returns nested_budget
- cascade disabled + no nested_budget → returns AgentBudget() default
- cascade enabled + nested_budget + parent remaining sufficient → returns intersection min()
- cascade enabled + nested_budget + parent remaining tight → returns parent remaining (min enforcement)
- cascade enabled + no nested_budget → returns parent remaining directly
- no parent budget/state provided → returns AgentBudget() (standalone)
"""
import time

import pytest

from magic_llm.agent._loop_shared import _compute_child_budget
from magic_llm.agent.types import (
    AgentBudget,
    AgentState,
    TaskManifest,
)


def _make_manifest(
    id: str = "test_task",
    budget_cascade: bool = False,
    nested_budget: AgentBudget | None = None,
) -> TaskManifest:
    return TaskManifest(
        id=id,
        name="Test Task",
        description="A test task",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        budget_cascade=budget_cascade,
        nested_budget=nested_budget,
    )


class TestComputeChildBudget:
    """Child budget computation based on cascade policy."""

    def test_cascade_disabled_nested_budget_provided(self):
        """cascade disabled + nested_budget provided → returns nested_budget."""
        manifest = _make_manifest(
            budget_cascade=False,
            nested_budget=AgentBudget(max_iterations=10),
        )
        parent_budget = AgentBudget(max_iterations=20)
        parent_state = AgentState(step=5, start_time=time.monotonic())
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Child uses own budget exclusively
        assert result.max_iterations == 10

    def test_cascade_disabled_no_nested_budget(self):
        """cascade disabled + no nested_budget → returns AgentBudget() default."""
        manifest = _make_manifest(
            budget_cascade=False,
            nested_budget=None,
        )
        parent_budget = AgentBudget(max_iterations=20)
        parent_state = AgentState(step=5, start_time=time.monotonic())
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Child uses default budget
        assert result.max_iterations == 150  # AgentBudget default

    def test_cascade_enabled_nested_budget_parent_remaining_sufficient(self):
        """cascade enabled + nested_budget + parent remaining sufficient → returns intersection min()."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=AgentBudget(max_iterations=3),
        )
        parent_budget = AgentBudget(max_iterations=10)
        parent_state = AgentState(step=2, start_time=time.monotonic())  # 8 remaining
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Intersection: min(3, 8) = 3
        assert result.max_iterations == 3

    def test_cascade_enabled_nested_budget_parent_remaining_tight(self):
        """cascade enabled + nested_budget + parent remaining tight → returns parent remaining (min enforcement)."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=AgentBudget(max_iterations=5),
        )
        parent_budget = AgentBudget(max_iterations=10)
        parent_state = AgentState(step=8, start_time=time.monotonic())  # 2 remaining
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Intersection: min(5, 2) = 2 (parent remaining enforced)
        assert result.max_iterations == 2

    def test_cascade_enabled_no_nested_budget(self):
        """cascade enabled + no nested_budget → returns parent remaining directly."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=None,
        )
        parent_budget = AgentBudget(max_iterations=10)
        parent_state = AgentState(step=3, start_time=time.monotonic())  # 7 remaining
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Child inherits parent remaining budget
        assert result.max_iterations == 7

    def test_no_parent_budget_state_provided(self):
        """no parent budget/state provided → returns AgentBudget() (standalone)."""
        manifest = _make_manifest(
            budget_cascade=False,
            nested_budget=None,
        )
        
        result = _compute_child_budget(manifest, None, None)
        
        # Standalone execution uses default budget
        assert result.max_iterations == 150  # AgentBudget default

    def test_no_parent_with_nested_budget(self):
        """no parent + nested_budget provided → returns nested_budget."""
        manifest = _make_manifest(
            budget_cascade=False,
            nested_budget=AgentBudget(max_iterations=15),
        )
        
        result = _compute_child_budget(manifest, None, None)
        
        # Uses provided nested_budget
        assert result.max_iterations == 15

    def test_cascade_with_wall_clock_timeout(self):
        """cascade enabled computes wall_clock_timeout intersection."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=AgentBudget(
                max_iterations=5,
                wall_clock_timeout=10.0,
            ),
        )
        parent_budget = AgentBudget(
            max_iterations=10,
            wall_clock_timeout=15.0,
        )
        # Elapsed 5 seconds, remaining ~10 seconds
        start_time = time.monotonic() - 5.0
        parent_state = AgentState(step=3, start_time=start_time)
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Intersection: min(10.0, ~10.0) = ~10.0 (allow floating-point tolerance)
        assert result.wall_clock_timeout is not None
        assert abs(result.wall_clock_timeout - 10.0) < 0.1  # Tolerance for timing

    def test_cascade_with_token_limits(self):
        """cascade enabled computes token limits intersection."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=AgentBudget(
                max_iterations=5,
                max_input_tokens=1000,
                max_output_tokens=500,
            ),
        )
        parent_budget = AgentBudget(
            max_iterations=10,
            max_input_tokens=2000,
            max_output_tokens=1000,
        )
        parent_state = AgentState(
            step=3,
            total_input_tokens=1500,  # 500 remaining
            total_output_tokens=400,  # 600 remaining
            start_time=time.monotonic(),
        )
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Intersection: min(1000, 500) = 500 for input tokens
        # Intersection: min(500, 600) = 500 for output tokens
        assert result.max_input_tokens == 500
        assert result.max_output_tokens == 500

    def test_cascade_parent_remaining_zero_iterations(self):
        """cascade enabled when parent has no iterations remaining."""
        manifest = _make_manifest(
            budget_cascade=True,
            nested_budget=AgentBudget(max_iterations=5),
        )
        parent_budget = AgentBudget(max_iterations=10)
        parent_state = AgentState(step=10, start_time=time.monotonic())  # 0 remaining
        
        result = _compute_child_budget(manifest, parent_budget, parent_state)
        
        # Child gets 0 iterations (parent exhausted)
        assert result.max_iterations == 0
