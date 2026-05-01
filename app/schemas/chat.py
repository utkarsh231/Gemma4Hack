from datetime import datetime
from enum import StrEnum
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator

from app.schemas.notes import DetailLevel, SourceStats


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


class YouTubeSessionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    learner_goal: str | None = Field(default=None, max_length=500)
    detail_level: DetailLevel = DetailLevel.standard

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, value: str) -> str:
        stripped = value.strip()
        allowed_hosts = (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        )
        if not stripped.startswith(("https://", "http://")):
            raise ValueError("YouTube URL must start with http:// or https://.")

        parsed = urlparse(stripped)
        host = parsed.netloc.lower()
        if host not in allowed_hosts:
            raise ValueError("Only YouTube video URLs are supported.")
        if host.endswith("youtu.be") and not parsed.path.strip("/"):
            raise ValueError("YouTube short URL is missing a video id.")
        if "youtube.com" in host and parsed.path == "/watch" and not parse_qs(parsed.query).get("v"):
            raise ValueError("YouTube watch URL is missing a video id.")
        if "youtube.com" in host and not (
            parsed.path == "/watch" or parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/")
        ):
            raise ValueError("Only YouTube video URLs are supported.")
        return stripped
