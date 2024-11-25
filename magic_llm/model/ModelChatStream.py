from pydantic import BaseModel
from typing import Optional, List, Any


class DeltaModel(BaseModel):
    content: Optional[str] = ''
    role: Optional[str] = 'assistant'


class ChoiceModel(BaseModel):
    index: Optional[int] = 0
    delta: Optional[DeltaModel] = DeltaModel()
    logprobs: Optional[Any] = None
    finish_reason: Optional[Any] = None


class UsageModel(BaseModel):
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    ttft: Optional[float] = 0
    ttf: Optional[float] = 0
    tps: Optional[float] = 0


class ChatCompletionModel(BaseModel):
    id: str
    object: Optional['str'] = 'chat.completion.chunk'
    created: Optional[int] = None
    model: str
    system_fingerprint: Optional[Any] = None
    choices: List[ChoiceModel]
    usage: Optional[UsageModel] = UsageModel()


class ChatMetaModel(BaseModel):
    TTFB: float = 0
    TTF: float
    TPS: float
    status: str = 'success'
