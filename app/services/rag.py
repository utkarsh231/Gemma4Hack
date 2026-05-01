from dataclasses import dataclass
import math
import re

from app.core.config import Settings
from app.services.pdf_text import ExtractedPdf


class RagError(Exception):
    """Raised when document indexing or retrieval fails."""


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    page: int
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int | None
    score: float | None = None
    source: str = "semantic"


class PineconeRagService:
    text_field = "chunk_text"
    fallback_text_field = "text"

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def index_source(self, *, namespace: str, source: ExtractedPdf, material_id: str | None = None) -> None:
        index = self._index()
        chunks = chunk_extracted_pdf(
            source,
            chunk_chars=self.settings.rag_chunk_chars,
            overlap_chars=self.settings.rag_chunk_overlap_chars,
            id_prefix=material_id or namespace,
        )
        records = [
            {
                "_id": chunk.id,
                self.text_field: chunk.text,
                self.fallback_text_field: chunk.text,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "filename": source.filename,
                "material_id": material_id or namespace,
            }
            for chunk in chunks
        ]

        try:
            index.upsert_records(namespace=namespace, records=records)
        except Exception as exc:
            raise RagError("Could not index document chunks in Pinecone.") from exc

    def retrieve(self, *, namespace: str, question: str) -> list[RetrievedChunk]:
        index = self._index()
        try:
            results = index.search(
                namespace=namespace,
                query={"inputs": {"text": question}, "top_k": self.settings.rag_top_k},
                fields=[self.text_field, self.fallback_text_field, "page"],
            )
        except Exception as exc:
            raise RagError("Could not retrieve relevant document chunks from Pinecone.") from exc

        hits = _get_hits(results)
        return [
            RetrievedChunk(
                text=_hit_text(hit, self.text_field, self.fallback_text_field),
                page=_coerce_int(hit.get("fields", {}).get("page")),
                score=_coerce_float(hit.get("_score")),
                source="semantic",
            )
            for hit in hits
            if _hit_text(hit, self.text_field, self.fallback_text_field)
        ]

    def _index(self):
        if not self.settings.pinecone_api_key:
            raise RagError("PINECONE_API_KEY is not configured.")

        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RagError("The pinecone package is not installed.") from exc

        try:
            client = Pinecone(api_key=self.settings.pinecone_api_key)
            if not client.has_index(self.settings.pinecone_index_name):
                client.create_index_for_model(
                    name=self.settings.pinecone_index_name,
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                    embed={
                        "model": self.settings.pinecone_embedding_model,
                        "field_map": {"text": self.text_field},
                    },
                )
            return client.Index(self.settings.pinecone_index_name)
        except Exception as exc:
            raise RagError("Could not connect to the Pinecone index.") from exc


def chunk_extracted_pdf(
    source: ExtractedPdf,
    *,
    chunk_chars: int,
    overlap_chars: int,
    id_prefix: str | None = None,
) -> list[DocumentChunk]:
    if overlap_chars >= chunk_chars:
        overlap_chars = max(0, chunk_chars // 5)

    chunks: list[DocumentChunk] = []
    for page, page_text in _iter_pages(source.text):
        start = 0
        while start < len(page_text):
            end = min(start + chunk_chars, len(page_text))
            chunk_text = page_text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        id=f"{id_prefix or source.filename}:{len(chunks):05d}",
                        text=chunk_text,
                        page=page,
                        chunk_index=len(chunks),
                    )
                )
            if end == len(page_text):
                break
            start = max(0, end - overlap_chars)

    if not chunks:
        raise RagError("The document did not produce any chunks for retrieval.")
    return chunks


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant document chunks were retrieved."
    return "\n\n".join(
        f"[Page {chunk.page if chunk.page is not None else 'unknown'} | {chunk.source} score {chunk.score:.4f}]\n{chunk.text}"
        if chunk.score is not None
        else f"[Page {chunk.page if chunk.page is not None else 'unknown'} | {chunk.source}]\n{chunk.text}"
        for chunk in chunks
    )


def retrieve_keyword_chunks(
    *,
    source: ExtractedPdf,
    question: str,
    chunk_chars: int,
    overlap_chars: int,
    top_k: int,
) -> list[RetrievedChunk]:
    query_terms = _tokenize(question)
    if not query_terms:
        return []

    chunks = chunk_extracted_pdf(source, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    scored_chunks: list[RetrievedChunk] = []
    total_chunks = len(chunks)
    doc_freq: dict[str, int] = {
        term: sum(1 for chunk in chunks if term in set(_tokenize(chunk.text)))
        for term in set(query_terms)
    }

    for chunk in chunks:
        chunk_terms = _tokenize(chunk.text)
        if not chunk_terms:
            continue

        term_counts = {term: chunk_terms.count(term) for term in set(query_terms)}
        score = 0.0
        for term, count in term_counts.items():
            if count == 0:
                continue
            inverse_doc_freq = math.log((total_chunks + 1) / (doc_freq.get(term, 0) + 1)) + 1
            score += count * inverse_doc_freq

        phrase_bonus = sum(1 for term in set(query_terms) if term in chunk.text.lower())
        score += phrase_bonus * 0.25

        if score > 0:
            scored_chunks.append(RetrievedChunk(text=chunk.text, page=chunk.page, score=score, source="keyword"))

    return sorted(scored_chunks, key=lambda chunk: chunk.score or 0, reverse=True)[:top_k]


def merge_retrieved_chunks(
    *,
    semantic_chunks: list[RetrievedChunk],
    keyword_chunks: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()

    for chunk in [*semantic_chunks, *keyword_chunks]:
        key = _dedupe_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break

    return merged


def _iter_pages(text: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r"\[Page (?P<page>\d+)\]\n", text))
    if not matches:
        return [(1, text.strip())]

    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        page = int(match.group("page"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append((page, page_text))
    return pages


def _get_hits(results) -> list[dict]:
    if hasattr(results, "to_dict"):
        results = results.to_dict()
    if isinstance(results, dict):
        return results.get("result", {}).get("hits", [])
    result = getattr(results, "result", None)
    if isinstance(result, dict):
        return result.get("hits", [])
    hits = getattr(result, "hits", None)
    if isinstance(hits, list):
        return hits
    return []


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hit_text(hit: dict, *field_names: str) -> str:
    fields = hit.get("fields", {})
    for field_name in field_names:
        text = str(fields.get(field_name, "")).strip()
        if text:
            return text
    return ""


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _dedupe_key(chunk: RetrievedChunk) -> str:
    return f"{chunk.page}:{chunk.text[:160]}"
