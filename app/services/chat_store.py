from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from app.schemas.chat import ChatMessage, ChatRole
from app.schemas.notes import SourceStats
from app.services.pdf_text import ExtractedPdf
from app.services.xp import calculate_level, calculate_next_level_xp, calculate_session_xp, calculate_tier


@dataclass
class UserXpSummary:
    total_xp: int = 0
    current_level: int = 1
    current_tier: str = "sprout"
    current_tier_display_name: str = "Sprout"
    next_level_xp: int = 250
    completed_tracks: int = 0
    total_focus_seconds: int = 0


@dataclass
class SessionXpBreakdown:
    session_completion_xp: int
    focus_time_xp: int
    quiz_completion_xp: int
    milestone_bonus_xp: int


@dataclass
class ChatSession:
    id: str
    rag_namespace: str
    material_id: str | None
    source: ExtractedPdf
    source_stats: SourceStats
    status: str
    started_at: datetime
    ended_at: datetime | None = None
    actual_duration_seconds: int | None = None
    quiz_completed: bool = False
    xp_awarded: int = 0
    xp_awarded_at: datetime | None = None
    xp_breakdown: SessionXpBreakdown | None = None
    messages: list[ChatMessage] = field(default_factory=list)


class InMemoryChatStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._xp_summary = UserXpSummary()
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
            status="active",
            started_at=datetime.now(UTC),
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
            session.quiz_completed = True
            if session.messages:
                session.messages[0] = message
            else:
                session.messages.append(message)
        return message

    def complete_session(
        self,
        *,
        session_id: str,
        actual_duration_seconds: int | None = None,
    ) -> tuple[ChatSession, UserXpSummary] | None:
        completed_at = datetime.now(UTC)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.status == "completed":
                return session, self._xp_summary

            duration_seconds = actual_duration_seconds
            if duration_seconds is None:
                duration_seconds = max(0, int((completed_at - session.started_at).total_seconds()))

            completed_track_count = self._xp_summary.completed_tracks + 1
            breakdown = calculate_session_xp(
                actual_duration_seconds=duration_seconds,
                quiz_completed=session.quiz_completed,
                completed_track_count=completed_track_count,
            )

            session.status = "completed"
            session.ended_at = completed_at
            session.actual_duration_seconds = duration_seconds
            session.xp_awarded = breakdown.total
            session.xp_awarded_at = completed_at
            session.xp_breakdown = SessionXpBreakdown(
                session_completion_xp=breakdown.session_completion_xp,
                focus_time_xp=breakdown.focus_time_xp,
                quiz_completion_xp=breakdown.quiz_completion_xp,
                milestone_bonus_xp=breakdown.milestone_bonus_xp,
            )

            self._xp_summary.total_xp += breakdown.total
            self._xp_summary.completed_tracks = completed_track_count
            self._xp_summary.total_focus_seconds += duration_seconds
            self._refresh_xp_summary_locked()
            return session, self._xp_summary

    def get_xp_summary(self) -> UserXpSummary:
        with self._lock:
            return self._xp_summary

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._xp_summary = UserXpSummary()

    def _refresh_xp_summary_locked(self) -> None:
        tier = calculate_tier(self._xp_summary.total_xp)
        self._xp_summary.current_level = calculate_level(self._xp_summary.total_xp)
        self._xp_summary.current_tier = tier.key
        self._xp_summary.current_tier_display_name = tier.display_name
        self._xp_summary.next_level_xp = calculate_next_level_xp(self._xp_summary.total_xp)

    @staticmethod
    def _build_message(*, role: ChatRole, content_markdown: str) -> ChatMessage:
        return ChatMessage(
            id=str(uuid4()),
            role=role,
            content_markdown=content_markdown.strip(),
            created_at=datetime.now(UTC),
        )


chat_store = InMemoryChatStore()
