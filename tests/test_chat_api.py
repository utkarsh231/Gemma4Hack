from collections.abc import Iterator
import pytest
from fastapi.testclient import TestClient

from app.api.routes import chat as chat_routes
from app.api.routes.chat import get_notes_service, get_rag_service
from app.main import app
from app.schemas.notes import DetailLevel
from app.services.chat_store import chat_store
from app.services.gemma_notes import build_notes_response
from app.services.pdf_text import ExtractedPdf
from app.services.rag import RetrievedChunk


class FakeNotesService:
    def generate_notes(
        self,
        extracted_pdf: ExtractedPdf,
        learner_goal: str | None,
        detail_level: DetailLevel,
    ):
        return build_notes_response(
            notes_markdown="### **Generated Notes**\n\n**Overview:**\nThis is a test note.",
            source=extracted_pdf,
        )

    def generate_notes_from_youtube(
        self,
        *,
        youtube_url: str,
        learner_goal: str | None,
        detail_level: DetailLevel,
    ):
        source = ExtractedPdf(
            filename=youtube_url,
            text="### **Video Notes**\n\n[00:00] This video explains attention and learning.",
            page_count=1,
            extracted_characters=71,
            truncated=False,
        )
        return build_notes_response(
            notes_markdown=source.text,
            source=source,
        ), source

    def answer_question(
        self,
        *,
        source: ExtractedPdf,
        notes_markdown: str,
        conversation_markdown: str,
        retrieved_context: str,
        question: str,
    ) -> str:
        return f"**Answer:** You asked: {question}\n\n{retrieved_context}"


class FakeRagService:
    def index_source(self, *, session_id: str, source: ExtractedPdf) -> None:
        return None

    def retrieve(self, *, session_id: str, question: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(text="Attention improves with small study blocks.", page=1, score=0.9)]


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        chat_routes,
        "extract_pdf_text",
        lambda raw_pdf, filename, max_pages, max_chars: ExtractedPdf(
            filename=filename,
            text="This document explains attention and learning.",
            page_count=1,
            extracted_characters=47,
            truncated=False,
        ),
    )
    app.dependency_overrides[get_notes_service] = lambda: FakeNotesService()
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    yield
    app.dependency_overrides.clear()
    chat_store._sessions.clear()


def test_create_chat_session_and_ask_question() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/chat/sessions/from-pdf",
        files={"file": ("lesson.pdf", b"%PDF-1.4 fake test bytes", "application/pdf")},
        data={"learner_goal": "Study", "detail_level": "standard"},
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["session_id"]
    assert created["messages"][0]["role"] == "assistant"
    assert "Generated Notes" in created["messages"][0]["content_markdown"]

    message_response = client.post(
        f"/api/v1/chat/sessions/{created['session_id']}/messages",
        json={"message": "What should I remember first?"},
    )

    assert message_response.status_code == 200
    message = message_response.json()["message"]
    assert message["role"] == "assistant"
    assert "What should I remember first?" in message["content_markdown"]
    assert "Attention improves with small study blocks." in message["content_markdown"]


def test_create_chat_session_from_youtube() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/chat/sessions/from-youtube",
        json={
            "url": "https://www.youtube.com/watch?v=abc123",
            "learner_goal": "Study",
            "detail_level": "standard",
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["session_id"]
    assert created["source_stats"]["filename"] == "https://www.youtube.com/watch?v=abc123"
    assert "Video Notes" in created["messages"][0]["content_markdown"]
