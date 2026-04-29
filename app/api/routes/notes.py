import logging
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.notes import DetailLevel, NotesResponse
from app.services.gemma_notes import GemmaNotesService, NotesGenerationError
from app.services.pdf_text import PdfExtractionError, extract_pdf_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notes", tags=["notes"])


def get_notes_service(settings: Annotated[Settings, Depends(get_settings)]) -> GemmaNotesService:
    return GemmaNotesService(settings=settings)


@router.post("/from-pdf", response_model=NotesResponse, status_code=status.HTTP_201_CREATED)
async def create_notes_from_pdf(
    file: Annotated[UploadFile, File(description="PDF study material to transform into ADHD-friendly notes.")],
    learner_goal: Annotated[str | None, Form(max_length=500)] = None,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.standard,
    settings: Annotated[Settings, Depends(get_settings)] = None,
    notes_service: Annotated[GemmaNotesService, Depends(get_notes_service)] = None,
) -> NotesResponse:
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
        extracted = extract_pdf_text(
            raw_pdf,
            filename=file.filename or "uploaded.pdf",
            max_pages=settings.max_pdf_pages,
            max_chars=settings.max_extracted_chars,
        )
    except PdfExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    try:
        return await anyio.to_thread.run_sync(
            notes_service.generate_notes,
            extracted,
            learner_goal,
            detail_level,
        )
    except NotesGenerationError as exc:
        logger.exception("notes_generation_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
