import logging
from functools import partial
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ChatRole, ChatSessionResponse, LinkSessionRequest, YouTubeSessionRequest
from app.schemas.notes import DetailLevel
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def get_notes_service(settings: Annotated[Settings, Depends(get_settings)]) -> GemmaNotesService:
    return GemmaNotesService(settings=settings)


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> PineconeRagService:
    return PineconeRagService(settings=settings)


@router.post("/sessions/from-pdf", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_pdf(
    file: Annotated[UploadFile, File(description="PDF study material to transform into a chat session.")],
    learner_goal: Annotated[str | None, Form(max_length=500)] = None,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.standard,
    session_id: Annotated[str | None, Form(max_length=100)] = None,
    material_id: Annotated[str | None, Form(max_length=100)] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)] = None,
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)] = None,
) -> ChatSessionResponse:
    extracted = await _extract_uploaded_pdf(file=file, settings=settings)

    try:
        notes = await anyio.to_thread.run_sync(
            notes_service.generate_notes,
            extracted,
            learner_goal,
            detail_level,
        )
    except NotesGenerationError as exc:
        logger.exception("chat_session_notes_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    session = chat_store.create_session(
        source=extracted,
        initial_notes_markdown=notes.notes_markdown,
        session_id=session_id,
        material_id=material_id,
    )
    try:
        await anyio.to_thread.run_sync(partial(_index_source, rag_service=rag_service, session=session, source=extracted))
    except RagError as exc:
        chat_store.delete_session(session.id)
        logger.exception("chat_session_rag_indexing_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return _session_response(session)


@router.post("/sessions/from-youtube", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_youtube(
    payload: YouTubeSessionRequest,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)],
) -> ChatSessionResponse:
    try:
        notes, source = await anyio.to_thread.run_sync(
            partial(
                notes_service.generate_notes_from_youtube,
                youtube_url=payload.url,
                learner_goal=payload.learner_goal,
                detail_level=payload.detail_level,
            )
        )
    except NotesGenerationError as exc:
        logger.exception("chat_session_youtube_notes_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    session = chat_store.create_session(
        source=source,
        initial_notes_markdown=notes.notes_markdown,
        session_id=payload.session_id,
        material_id=payload.material_id,
    )
    try:
        await anyio.to_thread.run_sync(partial(_index_source, rag_service=rag_service, session=session, source=source))
    except RagError as exc:
        chat_store.delete_session(session.id)
        logger.exception("chat_session_youtube_rag_indexing_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return _session_response(session)


@router.post("/sessions/from-link", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session_from_link(
    payload: LinkSessionRequest,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)],
    rag_service: Annotated[PineconeRagService, Depends(get_rag_service)],
) -> ChatSessionResponse:
    try:
        notes, source = await anyio.to_thread.run_sync(
            partial(
                notes_service.generate_notes_from_article,
                url=payload.url,
                learner_goal=payload.learner_goal,
                detail_level=payload.detail_level,
            )
        )
    except NotesGenerationError as exc:
        logger.exception("chat_session_link_notes_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    session = chat_store.create_session(
        source=source,
        initial_notes_markdown=notes.notes_markdown,
        session_id=payload.session_id,
        material_id=payload.material_id,
    )
    try:
        await anyio.to_thread.run_sync(partial(_index_source, rag_service=rag_service, session=session, source=source))
    except RagError as exc:
        chat_store.delete_session(session.id)
        logger.exception("chat_session_link_rag_indexing_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return _session_response(session)


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
        messages=session.messages,
    )


def _conversation_markdown(session: ChatSession) -> str:
    recent_messages = session.messages[-8:]
    return "\n\n".join(
        f"{message.role.value.upper()}:\n{message.content_markdown}" for message in recent_messages
    )


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
