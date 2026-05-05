from enum import StrEnum

from pydantic import BaseModel, Field


class DetailLevel(StrEnum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class SourceStats(BaseModel):
    filename: str
    page_count: int = Field(ge=1)
    extracted_characters: int = Field(ge=1)
    truncated: bool


class RecommendedVideo(BaseModel):
    video_id: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=300)
    channel_title: str = Field(min_length=1, max_length=160)
    url: str = Field(min_length=1, max_length=500)
    embed_url: str = Field(min_length=1, max_length=500)
    thumbnail_url: str | None = Field(default=None, max_length=1000)
    search_query: str = Field(min_length=1, max_length=300)


class NotesResponse(BaseModel):
    notes_markdown: str = Field(min_length=1)
    source_stats: SourceStats
    recommended_videos: list[RecommendedVideo] = Field(default_factory=list, max_length=8)
