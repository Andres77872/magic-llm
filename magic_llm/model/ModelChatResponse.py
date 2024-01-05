from pydantic import BaseModel


class ModelChatResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    role: str
