from pydantic import BaseModel

from magic_llm.model.ModelChatStream import UsageModel


class ModelChatResponse(BaseModel):
    content: str
    usage: UsageModel
    role: str
