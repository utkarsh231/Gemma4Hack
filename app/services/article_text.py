from dataclasses import dataclass
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


class ArticleExtractionError(Exception):
    """Raised when an online article cannot be converted into usable text."""


@dataclass(frozen=True)
class ExtractedArticle:
    url: str
    title: str
    text: str
    extracted_characters: int
    truncated: bool


def extract_article_text(url: str, *, max_chars: int) -> ExtractedArticle:
    normalized_url = _validate_article_url(url)

    try:
        response = httpx.get(
            normalized_url,
            follow_redirects=True,
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Gemma4HackStudyBot/0.1)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ArticleExtractionError("Could not fetch the article URL.") from exc

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ArticleExtractionError("The URL did not return an HTML article page.")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
        tag.decompose()

    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "") or normalized_url
    content_root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [_clean_text(node.get_text(" ", strip=True)) for node in content_root.find_all(["h1", "h2", "h3", "p", "li"])]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)

    if len(text) < 100:
        text = _clean_text(content_root.get_text(" ", strip=True))

    if len(text) < 100:
        raise ArticleExtractionError("The article did not contain enough readable text.")

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()

    return ExtractedArticle(
        url=normalized_url,
        title=title,
        text=text,
        extracted_characters=len(text),
        truncated=truncated,
    )


def _validate_article_url(url: str) -> str:
    stripped = url.strip()
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ArticleExtractionError("Article URL must start with http:// or https://.")
    if parsed.netloc.lower() in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
        raise ArticleExtractionError("Use the YouTube source type for YouTube URLs.")
    return stripped


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
