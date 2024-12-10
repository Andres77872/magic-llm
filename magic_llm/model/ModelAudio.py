from typing import Optional
from pydantic import BaseModel


class AudioSpeechRequest(BaseModel):
    input: str
    model: str
    voice: str
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1

class AudioTranscriptionsRequest(BaseModel):
    file: bytes
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = 'json'
    temperature: Optional[float] = 0
