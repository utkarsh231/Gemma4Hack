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


class NotesResponse(BaseModel):
    notes_markdown: str = Field(min_length=1)
    source_stats: SourceStats
