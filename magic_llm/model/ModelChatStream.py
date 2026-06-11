from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[FunctionCall] = None


class DeltaModel(BaseModel):
    content: Optional[str] = ''
    role: Optional[str] = 'assistant'
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None
    annotations: Optional[List[Any]] = []
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[Any] = None


class ChoiceModel(BaseModel):
    index: Optional[int] = 0
    delta: Optional[DeltaModel] = DeltaModel()
    logprobs: Optional[Any] = None
    finish_reason: Optional[Any] = None


class PromptTokensDetailsModel(BaseModel):
    cached_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0


class CompletionsTokensDetailsModel(BaseModel):
    reasoning_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0
    accepted_prediction_tokens: Optional[int] = 0
    rejected_prediction_tokens: Optional[int] = 0


class UsageModel(BaseModel):
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokensDetailsModel] = Field(default_factory=PromptTokensDetailsModel)
    completion_tokens_details: Optional[CompletionsTokensDetailsModel] = Field(default_factory=CompletionsTokensDetailsModel)
    cached_tokens_write: Optional[int] = 0
    provider_request_id: Optional[str] = None
    service_tier: Optional[str] = None
    usage_source: Optional[str] = 'provider'
    provider_extra: Optional[Dict[str, Any]] = None
    attempt_index: Optional[int] = None
    attempt_status: Optional[str] = None
    ttft: Optional[float] = 0
    ttf: Optional[float] = 0
    tps: Optional[float] = 0


class ChatCompletionModel(BaseModel):
    id: str
    object: Optional[str] = 'chat.completion.chunk'
    created: Optional[float] = None
    model: str
    system_fingerprint: Optional[Any] = None
    choices: List[ChoiceModel]
    usage: Optional[UsageModel] = Field(default_factory=UsageModel)
    extras: Optional[Any] = None


class ChatMetaModel(BaseModel):
    TTFB: float = 0
    TTF: float
    TPS: float
    status: str = 'success'
