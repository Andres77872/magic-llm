from typing import Optional, List, Any

from pydantic import BaseModel


class DeltaModel(BaseModel):
    content: Optional[str] = ''
    role: Optional[str] = 'assistant'


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
    prompt_tokens_details: Optional[PromptTokensDetailsModel] = PromptTokensDetailsModel()
    completion_tokens_details: Optional[CompletionsTokensDetailsModel] = CompletionsTokensDetailsModel()
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
    usage: Optional[UsageModel] = UsageModel()
    extras: Optional[Any] = None


class ChatMetaModel(BaseModel):
    TTFB: float = 0
    TTF: float
    TPS: float
    status: str = 'success'
