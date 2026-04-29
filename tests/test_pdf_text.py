from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.services.pdf_text import PdfExtractionError, extract_pdf_text


def test_extract_pdf_text_rejects_empty_pdf() -> None:
    writer = PdfWriter()
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(PdfExtractionError):
        extract_pdf_text(buffer.getvalue(), filename="empty.pdf", max_pages=5, max_chars=1000)
