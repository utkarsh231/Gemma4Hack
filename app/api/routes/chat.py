import logging
from functools import partial
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.auth import AuthenticatedUser, get_optional_current_user
from app.core.config import Settings, get_settings
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatRole,
    ChatSessionResponse,
    DiagnosticQuizCreateRequest,
    DiagnosticQuizResponse,
    DiagnosticQuizSubmitRequest,
    FocusedNotesResponse,
    LinkSessionRequest,
    SessionCompleteRequest,
    SessionCompleteResponse,
    SourceSectionsResponse,
    XpBreakdownResponse,
    XpSummaryResponse,
    YouTubeSessionRequest,
)
from app.schemas.notes import DetailLevel
from app.services.article_text import ArticleExtractionError, extract_article_text
from app.services.chat_store import ChatSession, chat_store
from app.services.gemma_notes import GemmaNotesService, NotesGenerationError
from app.services.pdf_text import ExtractedPdf
from app.services.pdf_text import PdfExtractionError, extract_pdf_text
from app.services.rag import (
    PineconeRagService,
    RagError,
    format_retrieved_context,
    merge_retrieved_chunks,
    retrieve_keyword_chunks,
)
from app.services.supabase_xp import SupabaseXpError, SupabaseXpStore
from app.services.youtube_text import YouTubeExtractionError, extract_youtube_transcript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

PENDING_NOTES_MARKDOWN = (
    "### Preparing your adaptive notes\n\n"
    "FocusPath has read your source. Take the warm-up quiz first, then Gemma will build notes around your answers and confidence."
)


def get_notes_service(settings: Annotated[Settings, Depends(get_settings)]) -> GemmaNotesService:
    return GemmaNotesService(settings=settings)


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> PineconeRagService:
    return PineconeRagService(settings=settings)


async def _create_source_only_session(
    *,
    source: ExtractedPdf,
    session_id: str | None,
    material_id: str | None,
    rag_service: PineconeRagService,
    indexing_log_name: str,
) -> ChatSessionResponse:
    session = chat_store.create_session(
        source=source,
        initial_notes_markdown=PENDING_NOTES_MARKDOWN,
        session_id=session_id,
        material_id=material_id,
    )
    try:
        await anyio.to_thread.run_sync(partial(_index_source, rag_service=rag_service, session=session, source=source))
    except RagError as exc:
        chat_store.delete_session(session.id)
        logger.exception(indexing_log_name)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return _session_response(session)


@router.post("/sessions/from-pdf", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_pdf(
    file: Annotated[UploadFile, File(description="PDF study material to transform into a chat session.")],
    learner_goal: Annotated[str | None, Form(max_length=500)] = None,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.standard,
    session_id: Annotated[str | None, Form(max_length=100)] = None,
    material_id: Annotated[str | None, Form(max_length=100)] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)] = None,
) -> ChatSessionResponse:
    extracted = await _extract_uploaded_pdf(file=file, settings=settings)
    return await _create_source_only_session(
        source=extracted,
        session_id=session_id,
        material_id=material_id,
        rag_service=rag_service,
        indexing_log_name="chat_session_rag_indexing_failed",
    )


@router.post("/sessions/from-youtube", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_youtube(
    payload: YouTubeSessionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)],
) -> ChatSessionResponse:
    try:
        transcript = await anyio.to_thread.run_sync(
            partial(extract_youtube_transcript, payload.url, max_chars=settings.max_extracted_chars)
        )
    except YouTubeExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    source = ExtractedPdf(
        filename=payload.url,
        text=transcript.text,
        page_count=1,
        extracted_characters=transcript.extracted_characters,
        truncated=transcript.truncated,
    )
    return await _create_source_only_session(
        source=source,
        session_id=payload.session_id,
        material_id=payload.material_id,
        rag_service=rag_service,
        indexing_log_name="chat_session_youtube_rag_indexing_failed",
    )


@router.post("/sessions/from-link", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_link(
    payload: LinkSessionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)],
) -> ChatSessionResponse:
    try:
        article = await anyio.to_thread.run_sync(
            partial(extract_article_text, payload.url, max_chars=settings.max_extracted_chars)
        )
    except ArticleExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    source = ExtractedPdf(
        filename=article.url,
        text=f"{article.title}\n\n{article.text}",
        page_count=1,
        extracted_characters=article.extracted_characters,
        truncated=article.truncated,
    )
    return await _create_source_only_session(
        source=source,
        session_id=payload.session_id,
        material_id=payload.material_id,
        rag_service=rag_service,
        indexing_log_name="chat_session_link_rag_indexing_failed",
    )


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def create_chat_message(
    session_id: str,
    payload: ChatMessageRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)],
) -> ChatMessageResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        if not payload.material_id or not payload.notes_markdown:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
        return await _answer_restored_session(
            session_id=session_id,
            payload=payload,
            settings=settings,
            notes_service=notes_service,
            rag_service=rag_service,
        )

    user_message = chat_store.append_message(
        session_id=session_id,
        role=ChatRole.user,
        content_markdown=payload.message,
    )
    if user_message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    try:
        keyword_chunks = retrieve_keyword_chunks(
            source=session.source,
            question=payload.message,
            chunk_chars=settings.rag_chunk_chars,
            overlap_chars=settings.rag_chunk_overlap_chars,
            top_k=settings.rag_keyword_top_k,
        )
        semantic_chunks = await _retrieve_semantic_chunks(
            rag_service=rag_service,
            namespace=session.rag_namespace,
            question=payload.message,
            timeout_seconds=settings.rag_semantic_timeout_seconds,
        )
        chunks = merge_retrieved_chunks(
            semantic_chunks=semantic_chunks,
            keyword_chunks=keyword_chunks,
            limit=settings.rag_top_k,
        )
        answer = await anyio.to_thread.run_sync(
            partial(
                notes_service.answer_question,
                source=session.source,
                notes_markdown=session.messages[0].content_markdown,
                conversation_markdown=_conversation_markdown(session),
                retrieved_context=format_retrieved_context(chunks),
                question=payload.message,
            )
        )
    except NotesGenerationError as exc:
        logger.exception("chat_message_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    assistant_message = chat_store.append_message(
        session_id=session_id,
        role=ChatRole.assistant,
        content_markdown=answer,
    )
    if assistant_message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    return ChatMessageResponse(session_id=session_id, message=assistant_message)


@router.post("/sessions/{session_id}/sections", response_model=SourceSectionsResponse)
async def create_source_sections(
    session_id: str,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
    learner_goal: str | None = None,
) -> SourceSectionsResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    try:
        sections = await anyio.to_thread.run_sync(
            partial(notes_service.generate_source_sections, source=session.source, learner_goal=learner_goal)
        )
    except NotesGenerationError as exc:
        logger.exception("source_sections_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return SourceSectionsResponse(session_id=session_id, sections=sections)


@router.post("/sessions/{session_id}/quiz", response_model=DiagnosticQuizResponse)
async def create_diagnostic_quiz(
    session_id: str,
    payload: DiagnosticQuizCreateRequest,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
) -> DiagnosticQuizResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    try:
        questions = await anyio.to_thread.run_sync(
            partial(
                notes_service.generate_diagnostic_quiz,
                sections=payload.sections,
                learner_goal=payload.learner_goal,
            )
        )
    except NotesGenerationError as exc:
        logger.exception("diagnostic_quiz_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return DiagnosticQuizResponse(session_id=session_id, questions=questions)


@router.post("/sessions/{session_id}/quiz-results", response_model=FocusedNotesResponse)
async def create_focused_notes_from_quiz(
    session_id: str,
    payload: DiagnosticQuizSubmitRequest,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
) -> FocusedNotesResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    try:
        notes = await anyio.to_thread.run_sync(
            partial(
                notes_service.generate_focused_notes,
                source=session.source,
                learner_goal=payload.learner_goal,
                detail_level=payload.detail_level,
                quiz_markdown=_quiz_results_markdown(payload),
            )
        )
    except NotesGenerationError as exc:
        logger.exception("focused_notes_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    updated_message = chat_store.replace_initial_notes(session_id=session_id, notes_markdown=notes.notes_markdown)
    if updated_message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    updated_session = chat_store.get_session(session_id)
    if updated_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    return FocusedNotesResponse(
        session_id=session_id,
        notes_markdown=notes.notes_markdown,
        recommended_videos=notes.recommended_videos,
        chat_session=_session_response(updated_session),
    )


@router.post("/sessions/{session_id}/complete", response_model=SessionCompleteResponse)
async def complete_chat_session(
    session_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    current_user: Annotated[AuthenticatedUser | None, Depends(get_optional_current_user)],
    payload: SessionCompleteRequest | None = None,
) -> SessionCompleteResponse:
    completed = chat_store.complete_session(
        session_id=session_id,
        actual_duration_seconds=payload.actual_duration_seconds if payload else None,
    )
    if completed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    session, xp_summary = completed
    xp_earned = session.xp_awarded

    if current_user is not None and settings.supabase_url:
        if not settings.supabase_service_role_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_SERVICE_ROLE_KEY is required to persist XP.",
            )
        try:
            persisted = await SupabaseXpStore(settings=settings).complete_session(
                user_id=current_user.id,
                session_id=session.id,
                source_title=session.source_stats.filename,
                actual_duration_seconds=session.actual_duration_seconds or 0,
                xp_awarded=session.xp_awarded,
                xp_breakdown=session.xp_breakdown,
            )
        except SupabaseXpError as exc:
            logger.exception("supabase_xp_persistence_failed")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        xp_earned = persisted.xp_awarded
        xp_summary = persisted.xp_summary

    return SessionCompleteResponse(
        session_id=session.id,
        xp_earned=xp_earned,
        xp_breakdown=_xp_breakdown_response(session),
        xp_summary=_xp_summary_response(xp_summary),
        chat_session=_session_response(session),
    )


@router.get("/xp", response_model=XpSummaryResponse)
async def get_xp_summary(
    settings: Annotated[Settings, Depends(get_settings)],
    current_user: Annotated[AuthenticatedUser | None, Depends(get_optional_current_user)],
) -> XpSummaryResponse:
    if current_user is not None and settings.supabase_url:
        if not settings.supabase_service_role_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_SERVICE_ROLE_KEY is required to read persisted XP.",
            )
        try:
            xp_summary = await SupabaseXpStore(settings=settings).get_xp_summary(user_id=current_user.id)
        except SupabaseXpError as exc:
            logger.exception("supabase_xp_summary_fetch_failed")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return _xp_summary_response(xp_summary)

    return _xp_summary_response(chat_store.get_xp_summary())


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str) -> ChatSessionResponse:
    session = chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return _session_response(session)


async def _extract_uploaded_pdf(*, file: UploadFile, settings: Settings):
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF uploads are supported.")

    raw_pdf = await file.read()
    if not raw_pdf:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded PDF is empty.")
    if len(raw_pdf) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the {settings.max_upload_mb} MB upload limit.",
        )

    try:
        return extract_pdf_text(
            raw_pdf,
            filename=file.filename or "uploaded.pdf",
            max_pages=settings.max_pdf_pages,
            max_chars=settings.max_extracted_chars,
        )
    except PdfExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def _session_response(session: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        session_id=session.id,
        source_stats=session.source_stats,
        status=session.status,
        started_at=session.started_at,
        ended_at=session.ended_at,
        actual_duration_seconds=session.actual_duration_seconds,
        xp_awarded=session.xp_awarded,
        xp_awarded_at=session.xp_awarded_at,
        messages=session.messages,
    )


def _xp_summary_response(xp_summary) -> XpSummaryResponse:
    return XpSummaryResponse(
        total_xp=xp_summary.total_xp,
        current_level=xp_summary.current_level,
        current_tier=xp_summary.current_tier,
        current_tier_display_name=xp_summary.current_tier_display_name,
        next_level_xp=xp_summary.next_level_xp,
        completed_tracks=xp_summary.completed_tracks,
        total_focus_seconds=xp_summary.total_focus_seconds,
    )


def _xp_breakdown_response(session: ChatSession) -> XpBreakdownResponse | None:
    if session.xp_breakdown is None:
        return None
    return XpBreakdownResponse(
        session_completion_xp=session.xp_breakdown.session_completion_xp,
        focus_time_xp=session.xp_breakdown.focus_time_xp,
        quiz_completion_xp=session.xp_breakdown.quiz_completion_xp,
        milestone_bonus_xp=session.xp_breakdown.milestone_bonus_xp,
    )


def _conversation_markdown(session: ChatSession) -> str:
    recent_messages = session.messages[-8:]
    return "\n\n".join(
        f"{message.role.value.upper()}:\n{message.content_markdown}" for message in recent_messages
    )


def _quiz_results_markdown(payload: DiagnosticQuizSubmitRequest) -> str:
    answer_by_question = {answer.question_id: answer for answer in payload.answers}
    lines = []

    for question in payload.questions:
        answer = answer_by_question.get(question.id)
        selected_option = None
        if answer and answer.selected_option_id:
            selected_option = next(
                (option for option in question.options if option.id == answer.selected_option_id),
                None,
            )
        correct_option = next(
            (option for option in question.options if option.id == question.correct_option_id),
            None,
        )
        is_correct = bool(answer and answer.selected_option_id == question.correct_option_id)
        confidence = answer.confidence if answer else 0
        needs_focus = (not is_correct) or confidence < 60

        lines.extend(
            [
                f"## {question.unit_title}",
                f"Question: {question.question}",
                f"Selected answer: {selected_option.text if selected_option else 'Not sure / no answer'}",
                f"Correct answer: {correct_option.text if correct_option else question.correct_option_id}",
                f"Result: {'correct' if is_correct else 'incorrect'}",
                f"Confidence: {confidence}%",
                f"Needs extra focus: {'yes' if needs_focus else 'no'}",
                f"Explanation: {question.explanation}",
                "",
            ]
        )

    return "\n".join(lines).strip()


async def _retrieve_semantic_chunks(
    *,
    rag_service: PineconeRagService,
    namespace: str,
    question: str,
    timeout_seconds: float,
):
    with anyio.move_on_after(timeout_seconds) as cancel_scope:
        try:
            return await anyio.to_thread.run_sync(
                partial(_retrieve_from_rag, rag_service=rag_service, namespace=namespace, question=question),
                abandon_on_cancel=True,
            )
        except RagError:
            logger.warning("chat_message_semantic_retrieval_failed", exc_info=True)
            return []

    if cancel_scope.cancel_called:
        logger.warning("chat_message_semantic_retrieval_timed_out")
    return []


def _index_source(*, rag_service: PineconeRagService, session: ChatSession, source) -> None:
    try:
        rag_service.index_source(namespace=session.rag_namespace, source=source, material_id=session.material_id)
    except TypeError:
        rag_service.index_source(session_id=session.id, source=source)


def _retrieve_from_rag(*, rag_service: PineconeRagService, namespace: str, question: str):
    try:
        return rag_service.retrieve(namespace=namespace, question=question)
    except TypeError:
        return rag_service.retrieve(session_id=namespace, question=question)


async def _answer_restored_session(
    *,
    session_id: str,
    payload: ChatMessageRequest,
    settings: Settings,
    notes_service: GemmaNotesService,
    rag_service: PineconeRagService,
) -> ChatMessageResponse:
    try:
        semantic_chunks = await _retrieve_semantic_chunks(
            rag_service=rag_service,
            namespace=payload.material_id or session_id,
            question=payload.message,
            timeout_seconds=settings.rag_semantic_timeout_seconds,
        )
        retrieved_context = format_retrieved_context(semantic_chunks)
        source = ExtractedPdf(
            filename=payload.material_id or "restored-material",
            text=retrieved_context,
            page_count=1,
            extracted_characters=max(1, len(retrieved_context)),
            truncated=False,
        )
        answer = await anyio.to_thread.run_sync(
            partial(
                notes_service.answer_question,
                source=source,
                notes_markdown=payload.notes_markdown or "",
                conversation_markdown="",
                retrieved_context=retrieved_context,
                question=payload.message,
            )
        )
    except NotesGenerationError as exc:
        logger.exception("restored_chat_message_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    message = chat_store._build_message(role=ChatRole.assistant, content_markdown=answer)
    return ChatMessageResponse(session_id=session_id, message=message)
