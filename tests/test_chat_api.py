from collections.abc import Iterator
import pytest
from fastapi.testclient import TestClient

from app.api.routes import chat as chat_routes
from app.api.routes.chat import get_notes_service, get_rag_service
from app.main import app
from app.schemas.notes import DetailLevel
from app.services.chat_store import chat_store
from app.services.pdf_text import ExtractedPdf
from app.services.rag import RetrievedChunk


class FakeNotesService:
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
    monkeypatch.setattr(
        chat_routes,
        "extract_youtube_transcript",
        lambda url, max_chars: type(
            "Transcript",
            (),
            {
                "text": "[00:00] This video explains attention and learning.",
                "extracted_characters": 52,
                "truncated": False,
            },
        )(),
    )
    monkeypatch.setattr(
        chat_routes,
        "extract_article_text",
        lambda url, max_chars: type(
            "Article",
            (),
            {
                "url": url,
                "title": "Article title",
                "text": "This article explains attention and learning.",
                "extracted_characters": 45,
                "truncated": False,
            },
        )(),
    )
    app.dependency_overrides[get_notes_service] = lambda: FakeNotesService()
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    yield
    app.dependency_overrides.clear()
    chat_store.reset()


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
    assert created["status"] == "active"
    assert created["xp_awarded"] == 0
    assert "Preparing your adaptive notes" in created["messages"][0]["content_markdown"]

    message_response = client.post(
        f"/api/v1/chat/sessions/{created['session_id']}/messages",
        json={"message": "What should I remember first?"},
    )

    assert message_response.status_code == 200
    message = message_response.json()["message"]
    assert message["role"] == "assistant"
    assert "What should I remember first?" in message["content_markdown"]
    assert "Attention improves with small study blocks." in message["content_markdown"]


def test_complete_chat_session_awards_xp_from_time_and_updates_summary() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/chat/sessions/from-pdf",
        files={"file": ("lesson.pdf", b"%PDF-1.4 fake test bytes", "application/pdf")},
        data={"learner_goal": "Study", "detail_level": "standard"},
    )
    session_id = create_response.json()["session_id"]

    complete_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/complete",
        json={"actual_duration_seconds": 25 * 60},
    )

    assert complete_response.status_code == 200
    completed = complete_response.json()
    assert completed["xp_earned"] == 100
    assert completed["xp_breakdown"] == {
        "session_completion_xp": 50,
        "focus_time_xp": 50,
        "quiz_completion_xp": 0,
        "milestone_bonus_xp": 0,
    }
    assert completed["chat_session"]["status"] == "completed"
    assert completed["chat_session"]["actual_duration_seconds"] == 1500
    assert completed["chat_session"]["xp_awarded"] == 100
    assert completed["xp_summary"]["total_xp"] == 100
    assert completed["xp_summary"]["completed_tracks"] == 1
    assert completed["xp_summary"]["total_focus_seconds"] == 1500

    summary_response = client.get("/api/v1/chat/xp")

    assert summary_response.status_code == 200
    assert summary_response.json()["total_xp"] == 100


def test_complete_chat_session_is_idempotent() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/chat/sessions/from-pdf",
        files={"file": ("lesson.pdf", b"%PDF-1.4 fake test bytes", "application/pdf")},
        data={"learner_goal": "Study", "detail_level": "standard"},
    )
    session_id = create_response.json()["session_id"]

    first_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/complete",
        json={"actual_duration_seconds": 60},
    )
    second_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/complete",
        json={"actual_duration_seconds": 60 * 60},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["xp_earned"] == 52
    assert second_response.json()["xp_earned"] == 52
    assert second_response.json()["xp_summary"]["total_xp"] == 52
    assert second_response.json()["xp_summary"]["completed_tracks"] == 1


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
    assert "Preparing your adaptive notes" in created["messages"][0]["content_markdown"]


def test_create_chat_session_from_link() -> None:
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/chat/sessions/from-link",
        json={
            "url": "https://example.com/article",
            "learner_goal": "Study",
            "detail_level": "standard",
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["session_id"]
    assert created["source_stats"]["filename"] == "https://example.com/article"
    assert "Preparing your adaptive notes" in created["messages"][0]["content_markdown"]
