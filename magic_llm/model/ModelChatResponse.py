from typing import Optional, Any, List

from pydantic import BaseModel

from magic_llm.model.ModelChatStream import UsageModel


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None
    annotations: Optional[List[Any]] = []


class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class UsageDetails(BaseModel):
    cached_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0
    reasoning_tokens: Optional[int] = 0
    accepted_prediction_tokens: Optional[int] = 0
    rejected_prediction_tokens: Optional[int] = 0


class ModelChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: UsageModel
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None

    # Convenience properties to maintain backwards compatibility
    @property
    def content(self) -> Optional[str]:
        return self.choices[0].message.content if self.choices else None

    @property
    def role(self) -> str:
        return self.choices[0].message.role if self.choices else 'assistant'

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        return self.choices[0].message.tool_calls if self.choices else None

    @property
    def finish_reason(self) -> Optional[str]:
        return self.choices[0].finish_reason if self.choices else None
