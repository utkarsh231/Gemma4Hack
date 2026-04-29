from app.services.pdf_text import ExtractedPdf
from app.services.rag import chunk_extracted_pdf, format_retrieved_context, merge_retrieved_chunks, retrieve_keyword_chunks, RetrievedChunk


def test_chunk_extracted_pdf_preserves_page_metadata() -> None:
    source = ExtractedPdf(
        filename="lesson.pdf",
        text="[Page 1]\n" + ("A" * 120) + "\n\n[Page 2]\n" + ("B" * 80),
        page_count=2,
        extracted_characters=220,
        truncated=False,
    )

    chunks = chunk_extracted_pdf(source, chunk_chars=50, overlap_chars=10)

    assert len(chunks) > 2
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert chunks[0].text == "A" * 50


def test_format_retrieved_context_includes_pages_and_scores() -> None:
    context = format_retrieved_context([RetrievedChunk(text="Relevant text", page=3, score=0.75)])

    assert "[Page 3 | semantic score 0.7500]" in context
    assert "Relevant text" in context


def test_retrieve_keyword_chunks_scores_matching_terms() -> None:
    source = ExtractedPdf(
        filename="lesson.pdf",
        text="[Page 1]\nPhotosynthesis uses sunlight and chlorophyll.\n\n[Page 2]\nGravity pulls objects downward.",
        page_count=2,
        extracted_characters=91,
        truncated=False,
    )

    chunks = retrieve_keyword_chunks(
        source=source,
        question="How does chlorophyll use sunlight?",
        chunk_chars=500,
        overlap_chars=0,
        top_k=1,
    )

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].source == "keyword"


def test_merge_retrieved_chunks_deduplicates_semantic_and_keyword_results() -> None:
    semantic = [RetrievedChunk(text="Same chunk", page=1, score=0.9, source="semantic")]
    keyword = [RetrievedChunk(text="Same chunk", page=1, score=2.0, source="keyword")]

    chunks = merge_retrieved_chunks(semantic_chunks=semantic, keyword_chunks=keyword, limit=5)

    assert chunks == semantic
