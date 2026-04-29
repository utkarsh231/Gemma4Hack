from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


class PdfExtractionError(Exception):
    """Raised when uploaded PDF content cannot be converted into usable text."""


@dataclass(frozen=True)
class ExtractedPdf:
    filename: str
    text: str
    page_count: int
    extracted_characters: int
    truncated: bool


def extract_pdf_text(raw_pdf: bytes, *, filename: str, max_pages: int, max_chars: int) -> ExtractedPdf:
    try:
        reader = PdfReader(BytesIO(raw_pdf))
    except Exception as exc:
        raise PdfExtractionError("Could not read the uploaded PDF.") from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise PdfExtractionError("The PDF does not contain any pages.")
    if page_count > max_pages:
        raise PdfExtractionError(f"The PDF has {page_count} pages, which exceeds the {max_pages} page limit.")

    parts: list[str] = []
    truncated = False
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise PdfExtractionError(f"Could not extract text from page {index}.") from exc

        clean_page_text = " ".join(page_text.split())
        if not clean_page_text:
            continue

        next_part = f"\n\n[Page {index}]\n{clean_page_text}"
        remaining = max_chars - sum(len(part) for part in parts)
        if remaining <= 0:
            truncated = True
            break
        if len(next_part) > remaining:
            parts.append(next_part[:remaining])
            truncated = True
            break
        parts.append(next_part)

    text = "".join(parts).strip()
    if len(text) < 100:
        raise PdfExtractionError("The PDF did not contain enough extractable text. Scanned PDFs need OCR first.")

    return ExtractedPdf(
        filename=filename,
        text=text,
        page_count=page_count,
        extracted_characters=len(text),
        truncated=truncated,
    )
