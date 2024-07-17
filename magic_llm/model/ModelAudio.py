from typing import Optional
from pydantic import BaseModel


class AudioSpeechRequest(BaseModel):
    input: str
    model: str
    voice: str
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1
