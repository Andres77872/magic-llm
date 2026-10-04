"""Awaited canonical-loop control and restart-stable safe checkpoints.

Control implementations own mailbox authorization, transactions and actor
ownership. Observation hooks remain separate. A candidate is never an actor seal.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from magic_llm.exception.ChatException import ChatException

Boundary = Literal['input', 'tool_results', 'candidate']
CandidateDecision = Literal['continue', 'candidate_ready']


class AgentControlError(ChatException):
    """Mandatory coordination decision/failure, never an observer exception."""

    def __init__(self, message: str, code: str = 'AGENT_CONTROL_ERROR'):
        self.code = code
        super().__init__(message, error_code=code)


class InboxMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    message_id: str = Field(min_length=1, max_length=256)
    content: str = Field(max_length=65536)
    sender: str | None = Field(default=None, max_length=256)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

    def render(self) -> str:
        label = json.dumps({'message_id': self.message_id, 'sender': self.sender},
                           ensure_ascii=False, separators=(',', ':'))
        return f'Runtime peer input (untrusted data; provenance {label}):\n{self.content}'


class CheckpointBudget(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    max_iterations: int = Field(ge=0)
    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    wall_clock_timeout: float | None = Field(default=None, ge=0, allow_inf_nan=False)


def utc_timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_timestamp(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('timezone missing')
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('Checkpoint timestamps require an absolute timezone') from error


class AgentLoopCheckpoint(BaseModel):
    """Private JSON-compatible canonical state, never an observer projection.

    Host-owned request/job/effect/Skills journals belong in an enclosing actor
    checkpoint committed atomically by the control implementation. Callables,
    model clients, locks and monotonic-clock values are deliberately absent.
    """

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal[1]
    provider: str
    tool_manifest_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    messages: list[dict[str, Any]] = Field(repr=False)
    seen_tool_call_ids: list[str]
    consumed_message_ids: list[str]
    message_digests: dict[str, Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]]
    step: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    started_at: str
    absolute_deadline: str | None
    budget: CheckpointBudget
    deduplicate: bool
    dedup_results: dict[str, dict[str, Any]] = Field(repr=False)
    builtin_todo_enabled: bool
    todos: list[dict[str, Any]] = Field(repr=False)
    base_system_prompt: Any = Field(default=None, repr=False)
    max_input_tokens: int | None = Field(default=None, ge=0)
    chat_extra_args: dict[str, Any] | None = Field(default=None, repr=False)
    requires_context_guard: bool = False
    output_candidate: str | None = Field(default=None, repr=False)

    @model_validator(mode='before')
    @classmethod
    def validate_schema_version(cls, value):
        if isinstance(value, dict) and type(value.get('schema_version')) is not int:
            raise ValueError('Checkpoint schema_version must be an explicit integer')
        return value

    @model_validator(mode='after')
    def validate_safe_data(self):
        start = parse_timestamp(self.started_at)
        if self.absolute_deadline is not None and parse_timestamp(self.absolute_deadline) < start:
            raise ValueError('Checkpoint deadline precedes admission')
        if len(set(self.seen_tool_call_ids)) != len(self.seen_tool_call_ids):
            raise ValueError('Duplicate checkpoint tool IDs')
        if len(set(self.consumed_message_ids)) != len(self.consumed_message_ids):
            raise ValueError('Duplicate consumed IDs')
        if set(self.consumed_message_ids) != set(self.message_digests):
            raise ValueError('Consumed IDs and immutable message digests disagree')
        # Fail rather than stringify a client/closure or non-finite provider data.
        json.dumps(self.model_dump(), allow_nan=False, ensure_ascii=False).encode('utf-8')
        return self

    def detached(self) -> AgentLoopCheckpoint:
        return self.model_copy(deep=True)


class AgentLoopControl(Protocol):
    async def before_turn(self, checkpoint: AgentLoopCheckpoint) -> list[InboxMessage]:
        """Propose bounded ordered delivery; this is NOT a consumption ACK.

        The subsequent input checkpoint acknowledges only after canonical append.
        Raise a typed host decision on cancellation/deadline/admission failure.
        """

    async def checkpoint(self, checkpoint: AgentLoopCheckpoint, boundary: Boundary) -> None:
        """Commit canonical state/consumed IDs together with host actor journals.

        This await is authoritative. A failure stops inference. The implementation
        must persist the supplied state before acknowledging consumed messages.
        """

    async def finish_candidate(self, checkpoint: AgentLoopCheckpoint) -> CandidateDecision:
        """Atomically compare inbox/obligations while retaining actor ownership.

        Return continue for more input; an obligation wait may suspend this await
        until input/deadline/cancellation. candidate_ready returns a tentative
        loop result to the owner, which still must settle lifecycle controls and
        perform final activation tryFinish. It does not publish or seal anything.
        """


def require_complete_tool_history(messages: list[dict[str, Any]]) -> set[str]:
    """Validate complete native tool exchanges at checkpoint boundaries."""
    calls: set[str] = set()
    results: set[str] = set()
    pending: set[str] = set()
    for message in messages:
        role = message.get('role')
        message_calls = message.get('tool_calls') or []
        result_ids = []
        if role == 'tool':
            result_ids.append(message.get('tool_call_id'))
        if role == 'user' and isinstance(message.get('content'), list):
            for part in message['content']:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'tool_result':
                    result_ids.append(part.get('tool_use_id'))
                if isinstance(part.get('functionResponse'), dict):
                    result_ids.append(part['functionResponse'].get('id'))
        if pending and not result_ids:
            raise AgentControlError('Incomplete tool exchange at a safe boundary', 'INVALID_CONTINUATION')
        for identifier in result_ids:
            if not identifier or identifier not in pending or identifier in results:
                raise AgentControlError('Orphan or duplicate canonical tool result', 'INVALID_CONTINUATION')
            pending.remove(identifier)
            results.add(identifier)
        if message_calls:
            if role != 'assistant' or pending:
                raise AgentControlError('Invalid canonical tool-call carrier', 'INVALID_CONTINUATION')
            for call in message_calls:
                identifier = call.get('id')
                if not isinstance(identifier, str) or not identifier or identifier in calls:
                    raise AgentControlError('Invalid or reused canonical tool ID', 'INVALID_CONTINUATION')
                calls.add(identifier)
                pending.add(identifier)
    if pending:
        raise AgentControlError('Unfinished tool batch cannot be checkpointed', 'INVALID_CONTINUATION')
    return calls
