from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.notes import SourceStats


class ChatRole(StrEnum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    id: str
    role: ChatRole
    content_markdown: str = Field(min_length=1)
    created_at: datetime


class ChatSessionResponse(BaseModel):
    session_id: str
    source_stats: SourceStats
    messages: list[ChatMessage] = Field(min_length=1)


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    session_id: str
    message: ChatMessage
