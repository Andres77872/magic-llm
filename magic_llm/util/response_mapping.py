"""Shared utilities for response normalization across providers."""

import time
from typing import Optional, List

from magic_llm.model.ModelChatResponse import (
    ModelChatResponse, Choice, Message, ToolCall, FunctionCall
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel, ChoiceModel, DeltaModel, UsageModel,
    ToolCall as StreamToolCall, FunctionCall as StreamFunctionCall
)


# ═══════════════════════════════════════════════════════════════════════
# FINISH REASON MAPPINGS
# ═══════════════════════════════════════════════════════════════════════

ANTHROPIC_FINISH_REASON_MAP = {
    'tool_use': 'tool_calls',
    'stop_sequence': 'stop',
    'max_tokens': 'length',
    'end_turn': 'stop'
}

GOOGLE_FINISH_REASON_MAP = {
    'STOP': 'stop',
    'MAX_TOKENS': 'length',
    'SAFETY': 'content_filter',
    'RECITATION': 'content_filter',
    'OTHER': 'stop',
    'FINISH_REASON_UNSPECIFIED': None
}

COHERE_FINISH_REASON_MAP = {
    'COMPLETE': 'stop',
    'MAX_TOKENS': 'length',
    'ERROR': 'stop',
    'ERROR_TOXIC': 'content_filter',
    'ERROR_LIMIT': 'length',
    'USER_CANCEL': 'stop'
}

AMAZON_FINISH_REASON_MAP = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'max_tokens': 'length',
    'content_filtered': 'content_filter',
    'tool_use': 'tool_calls',
    'FINISH': 'stop'
}


def map_finish_reason(
    provider_reason: Optional[str],
    mapping: dict,
    default: str = 'stop'
) -> Optional[str]:
    """Map provider-specific finish reason to OpenAI format."""
    if provider_reason is None:
        return None
    return mapping.get(provider_reason, default)


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_response(
    id: str,
    model: str,
    content: Optional[str],
    role: str = 'assistant',
    finish_reason: Optional[str] = 'stop',
    tool_calls: Optional[List[ToolCall]] = None,
    usage: Optional[UsageModel] = None,
    **extra
) -> ModelChatResponse:
    """Build a standardized ModelChatResponse."""

    message = Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        refusal=None,
        annotations=[]
    )

    choice = Choice(
        index=0,
        message=message,
        logprobs=extra.get('logprobs'),
        finish_reason=finish_reason
    )

    return ModelChatResponse(
        id=id,
        object='chat.completion',
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage or UsageModel(),
        service_tier=extra.get('service_tier'),
        system_fingerprint=extra.get('system_fingerprint')
    )


def build_stream_chunk(
    id: str,
    model: str,
    content: Optional[str] = '',
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[StreamToolCall]] = None,
    usage: Optional[UsageModel] = None,
    index: int = 0
) -> ChatCompletionModel:
    """Build a standardized ChatCompletionModel chunk."""

    delta = DeltaModel(
        content=content,
        role=role,
        tool_calls=tool_calls,
        refusal=None,
        annotations=[]
    )

    choice = ChoiceModel(
        index=index,
        delta=delta,
        finish_reason=finish_reason,
        logprobs=None
    )

    return ChatCompletionModel(
        id=id,
        object='chat.completion.chunk',
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage or UsageModel()
    )


def build_tool_call(
    id: str,
    name: str,
    arguments: str  # JSON string
) -> ToolCall:
    """Build a standardized ToolCall for non-streaming responses."""
    return ToolCall(
        id=id,
        type='function',
        function=FunctionCall(
            name=name,
            arguments=arguments
        )
    )


def build_stream_tool_call(
    id: str,
    name: Optional[str],
    arguments: str  # JSON string
) -> StreamToolCall:
    """Build a standardized ToolCall for streaming responses."""
    return StreamToolCall(
        id=id,
        type='function',
        function=StreamFunctionCall(
            name=name,
            arguments=arguments
        )
    )
