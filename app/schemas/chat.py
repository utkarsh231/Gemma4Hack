from datetime import datetime
from enum import StrEnum
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator

from app.schemas.notes import DetailLevel, RecommendedVideo, SourceStats


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
    status: str = "active"
    started_at: datetime
    ended_at: datetime | None = None
    actual_duration_seconds: int | None = Field(default=None, ge=0)
    xp_awarded: int = Field(default=0, ge=0)
    xp_awarded_at: datetime | None = None
    messages: list[ChatMessage] = Field(min_length=1)


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    material_id: str | None = Field(default=None, max_length=100)
    notes_markdown: str | None = Field(default=None, max_length=200_000)


class ChatMessageResponse(BaseModel):
    session_id: str
    message: ChatMessage


class SourceSection(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=1200)
    source_excerpt: str = Field(min_length=1, max_length=2000)


class SourceSectionsResponse(BaseModel):
    session_id: str
    sections: list[SourceSection] = Field(min_length=1, max_length=8)


class DiagnosticQuizOption(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=500)


class DiagnosticQuizQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    unit_title: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=500)
    options: list[DiagnosticQuizOption] = Field(min_length=3, max_length=4)
    correct_option_id: str = Field(min_length=1, max_length=20)
    explanation: str = Field(min_length=1, max_length=800)
    key_takeaway: str = Field(default="", max_length=800)
    study_note: str = Field(default="", max_length=1600)
    source_excerpt: str = Field(default="", max_length=1200)


class DiagnosticQuizResponse(BaseModel):
    session_id: str
    questions: list[DiagnosticQuizQuestion] = Field(min_length=1, max_length=8)


class DiagnosticQuizCreateRequest(BaseModel):
    learner_goal: str | None = Field(default=None, max_length=500)
    sections: list[SourceSection] = Field(min_length=1, max_length=8)


class DiagnosticQuizAnswer(BaseModel):
    question_id: str = Field(min_length=1, max_length=40)
    selected_option_id: str | None = Field(default=None, max_length=20)
    confidence: int = Field(ge=0, le=100)


class DiagnosticQuizSubmitRequest(BaseModel):
    learner_goal: str | None = Field(default=None, max_length=500)
    detail_level: DetailLevel = DetailLevel.standard
    answers: list[DiagnosticQuizAnswer] = Field(min_length=1, max_length=8)
    questions: list[DiagnosticQuizQuestion] = Field(min_length=1, max_length=8)


class FocusedNotesResponse(BaseModel):
    session_id: str
    notes_markdown: str = Field(min_length=1)
    recommended_videos: list[RecommendedVideo] = Field(default_factory=list, max_length=8)
    chat_session: ChatSessionResponse


class SessionCompleteRequest(BaseModel):
    actual_duration_seconds: int | None = Field(default=None, ge=0)


class XpBreakdownResponse(BaseModel):
    session_completion_xp: int = Field(ge=0)
    focus_time_xp: int = Field(ge=0)
    quiz_completion_xp: int = Field(ge=0)
    milestone_bonus_xp: int = Field(ge=0)


class XpSummaryResponse(BaseModel):
    total_xp: int = Field(ge=0)
    current_level: int = Field(ge=1)
    current_tier: str
    current_tier_display_name: str
    next_level_xp: int = Field(ge=0)
    completed_tracks: int = Field(ge=0)
    total_focus_seconds: int = Field(ge=0)


class SessionCompleteResponse(BaseModel):
    session_id: str
    xp_earned: int = Field(ge=0)
    xp_breakdown: XpBreakdownResponse | None = None
    xp_summary: XpSummaryResponse
    chat_session: ChatSessionResponse


class LinkSessionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    learner_goal: str | None = Field(default=None, max_length=500)
    detail_level: DetailLevel = DetailLevel.standard
    session_id: str | None = Field(default=None, max_length=100)
    material_id: str | None = Field(default=None, max_length=100)

    @field_validator("url")
    @classmethod
    def validate_link_url(cls, value: str) -> str:
        stripped = value.strip()
        parsed = urlparse(stripped)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Link URL must start with http:// or https://.")
        if parsed.netloc.lower() in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
            raise ValueError("Use the YouTube source type for YouTube URLs.")
        return stripped


class YouTubeSessionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    learner_goal: str | None = Field(default=None, max_length=500)
    detail_level: DetailLevel = DetailLevel.standard
    session_id: str | None = Field(default=None, max_length=100)
    material_id: str | None = Field(default=None, max_length=100)

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
