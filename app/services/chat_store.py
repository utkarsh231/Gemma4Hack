from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from app.schemas.chat import ChatMessage, ChatRole
from app.schemas.notes import SourceStats
from app.services.pdf_text import ExtractedPdf


@dataclass
class ChatSession:
    id: str
    rag_namespace: str
    material_id: str | None
    source: ExtractedPdf
    source_stats: SourceStats
    messages: list[ChatMessage] = field(default_factory=list)


class InMemoryChatStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = Lock()

    def create_session(
        self,
        *,
        source: ExtractedPdf,
        initial_notes_markdown: str,
        session_id: str | None = None,
        material_id: str | None = None,
    ) -> ChatSession:
        session_id = session_id or str(uuid4())
        source_stats = SourceStats(
            filename=source.filename,
            page_count=source.page_count,
            extracted_characters=source.extracted_characters,
            truncated=source.truncated,
        )
        initial_message = self._build_message(role=ChatRole.assistant, content_markdown=initial_notes_markdown)
        session = ChatSession(
            id=session_id,
            rag_namespace=material_id or session_id,
            material_id=material_id,
            source=source,
            source_stats=source_stats,
            messages=[initial_message],
        )

        with self._lock:
            self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def append_message(self, *, session_id: str, role: ChatRole, content_markdown: str) -> ChatMessage | None:
        message = self._build_message(role=role, content_markdown=content_markdown)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.messages.append(message)
        return message

    def replace_initial_notes(self, *, session_id: str, notes_markdown: str) -> ChatMessage | None:
        message = self._build_message(role=ChatRole.assistant, content_markdown=notes_markdown)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.messages:
                session.messages[0] = message
            else:
                session.messages.append(message)
        return message

    @staticmethod
    def _build_message(*, role: ChatRole, content_markdown: str) -> ChatMessage:
        return ChatMessage(
            id=str(uuid4()),
            role=role,
            content_markdown=content_markdown.strip(),
            created_at=datetime.now(UTC),
        )


chat_store = InMemoryChatStore()
