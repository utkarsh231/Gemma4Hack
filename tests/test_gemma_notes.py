import pytest

from app.schemas.notes import DetailLevel, NotesResponse
from app.services.gemma_notes import (
    NotesGenerationError,
    build_chat_prompt,
    build_notes_prompt,
    build_notes_response,
    build_youtube_notes_prompt,
)
from app.services.pdf_text import ExtractedPdf


@pytest.fixture
def source() -> ExtractedPdf:
    return ExtractedPdf(
        filename="lesson.pdf",
        text="A" * 200,
        page_count=2,
        extracted_characters=200,
        truncated=False,
    )


def test_build_notes_response_adds_source_stats(source: ExtractedPdf) -> None:
    raw = "### **Photosynthesis**\n\n**Overview:**\nPlants make food from light."

    parsed = build_notes_response(notes_markdown=raw, source=source)

    assert isinstance(parsed, NotesResponse)
    assert parsed.source_stats.filename == "lesson.pdf"
    assert parsed.notes_markdown.startswith("### **Photosynthesis**")


def test_build_notes_response_rejects_empty_notes(source: ExtractedPdf) -> None:
    with pytest.raises(NotesGenerationError):
        build_notes_response(notes_markdown="   ", source=source)


def test_build_notes_prompt_requests_markdown(source: ExtractedPdf) -> None:
    prompt = build_notes_prompt(extracted_pdf=source, learner_goal="Study", detail_level=DetailLevel.standard)

    assert "Return only the final notes in Markdown" in prompt
    assert "Quick Self-Check Questions" in prompt
    assert "Attention Breakpoints" in prompt


def test_build_youtube_notes_prompt_requests_timestamped_notes() -> None:
    source = ExtractedPdf(
        filename="https://www.youtube.com/watch?v=abc123",
        text="[00:00] Intro\n[01:23] Main idea",
        page_count=1,
        extracted_characters=28,
        truncated=False,
    )
    prompt = build_youtube_notes_prompt(source=source, learner_goal="Study", detail_level=DetailLevel.standard)

    assert "Create ADHD-friendly study notes from this YouTube transcript" in prompt
    assert "Timestamped focus blocks" in prompt
    assert "Use timestamps whenever possible" in prompt
    assert "[01:23] Main idea" in prompt


def test_build_chat_prompt_uses_retrieved_context_instead_of_full_source(source: ExtractedPdf) -> None:
    prompt = build_chat_prompt(
        source=source,
        notes_markdown="Generated notes",
        conversation_markdown="USER:\nQuestion",
        retrieved_context="[Page 1]\nRelevant chunk",
        question="What matters?",
    )

    assert "Retrieved document context:" in prompt
    assert "Relevant chunk" in prompt
    assert source.text not in prompt
