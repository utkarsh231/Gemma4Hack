import re
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.schemas.notes import RecommendedVideo
from app.services.pdf_text import ExtractedPdf

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

QUERY_SUFFIX = "beginner explanation tutorial"

STOPWORDS = {
    "about",
    "action",
    "adhd",
    "article",
    "attention",
    "block",
    "blocks",
    "breakpoint",
    "check",
    "deep",
    "definition",
    "definitions",
    "focus",
    "friendly",
    "important",
    "introduction",
    "key",
    "learner",
    "material",
    "memory",
    "notes",
    "overview",
    "points",
    "quick",
    "self",
    "short",
    "study",
    "summary",
    "takeaways",
    "this",
    "video",
    "what",
    "why",
}


class YouTubeVideoSearchError(Exception):
    """Raised when YouTube video recommendations cannot be fetched."""


HttpGet = Callable[..., httpx.Response]


def search_youtube_learning_videos(
    *,
    settings: Settings,
    source: ExtractedPdf,
    notes_markdown: str,
    learner_goal: str | None = None,
    max_videos: int | None = None,
    http_get: HttpGet | None = None,
) -> list[RecommendedVideo]:
    if not settings.youtube_api_key:
        return []

    target_count = settings.youtube_max_videos if max_videos is None else max_videos
    if target_count <= 0:
        return []

    queries = build_video_search_queries(
        source=source,
        notes_markdown=notes_markdown,
        learner_goal=learner_goal,
        max_queries=min(3, target_count),
    )
    if not queries:
        return []

    videos: list[RecommendedVideo] = []
    seen_video_ids: set[str] = set()
    get = http_get or httpx.get

    for query in queries:
        payload = fetch_youtube_search_payload(settings=settings, query=query, max_results=target_count, http_get=get)
        for item in payload.get("items", []):
            video = parse_youtube_search_item(item=item, search_query=query)
            if video is None or video.video_id in seen_video_ids:
                continue
            videos.append(video)
            seen_video_ids.add(video.video_id)
            if len(videos) >= target_count:
                return videos

    return videos


def append_recommended_videos_to_notes(notes_markdown: str, videos: list[RecommendedVideo]) -> str:
    cleaned_notes = notes_markdown.strip()
    if not videos:
        return cleaned_notes

    video_lines = ["## Recommended Videos", ""]
    for index, video in enumerate(videos, start=1):
        video_lines.extend(
            [
                f"{index}. [{video.title}]({video.url})",
                f"   * Channel: {video.channel_title}",
                f"   * Embed URL: {video.embed_url}",
            ]
        )
    return f"{cleaned_notes}\n\n{chr(10).join(video_lines)}"


def fetch_youtube_search_payload(
    *,
    settings: Settings,
    query: str,
    max_results: int,
    http_get: HttpGet,
) -> dict:
    try:
        response = http_get(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet",
                "type": "video",
                "videoEmbeddable": "true",
                "safeSearch": "strict",
                "order": "relevance",
                "maxResults": max(1, min(10, max_results)),
                "q": query,
                "key": settings.youtube_api_key,
            },
            timeout=8.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise YouTubeVideoSearchError("YouTube video search failed.") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise YouTubeVideoSearchError("YouTube video search returned an invalid payload.")
    return payload


def parse_youtube_search_item(*, item: dict, search_query: str) -> RecommendedVideo | None:
    item_id = item.get("id")
    snippet = item.get("snippet")
    if not isinstance(item_id, dict) or not isinstance(snippet, dict):
        return None

    video_id = item_id.get("videoId")
    title = clean_youtube_text(snippet.get("title"))
    channel_title = clean_youtube_text(snippet.get("channelTitle"))
    if not video_id or not title or not channel_title:
        return None

    thumbnails = snippet.get("thumbnails")
    thumbnail_url = choose_thumbnail_url(thumbnails if isinstance(thumbnails, dict) else {})

    return RecommendedVideo(
        video_id=video_id,
        title=title,
        channel_title=channel_title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        embed_url=f"https://www.youtube.com/embed/{video_id}",
        thumbnail_url=thumbnail_url,
        search_query=search_query,
    )


def build_video_search_queries(
    *,
    source: ExtractedPdf,
    notes_markdown: str,
    learner_goal: str | None,
    max_queries: int,
) -> list[str]:
    candidates: list[str] = []

    if learner_goal and learner_goal.strip():
        candidates.append(learner_goal.strip())

    candidates.extend(extract_markdown_headings(notes_markdown))
    candidates.extend(extract_source_title_candidates(source))
    candidates.extend(extract_keyword_phrases(notes_markdown))

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = clean_query_topic(candidate)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        queries.append(f"{cleaned} {QUERY_SUFFIX}")
        if len(queries) >= max_queries:
            break

    return queries


def extract_markdown_headings(markdown: str) -> list[str]:
    headings = []
    for match in re.finditer(r"^#{1,4}\s+(.+)$", markdown, flags=re.MULTILINE):
        heading = strip_markdown(match.group(1))
        if is_useful_topic(heading):
            headings.append(heading)
    return headings


def extract_source_title_candidates(source: ExtractedPdf) -> list[str]:
    candidates: list[str] = []
    first_line = next((line.strip() for line in source.text.splitlines() if line.strip()), "")
    if is_useful_topic(first_line):
        candidates.append(first_line)

    parsed = urlparse(source.filename)
    if parsed.scheme and parsed.netloc:
        slug = parsed.path.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
        if is_useful_topic(slug):
            candidates.append(slug)
    elif is_useful_topic(source.filename):
        candidates.append(source.filename.rsplit(".", 1)[0].replace("_", " "))

    return candidates


def extract_keyword_phrases(text: str) -> list[str]:
    cleaned = strip_markdown(text)
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", cleaned)
        if word.lower() not in STOPWORDS
    ]
    phrases: list[str] = []
    for index in range(max(0, len(words) - 2)):
        phrase = " ".join(words[index : index + 3])
        if is_useful_topic(phrase):
            phrases.append(phrase)
        if len(phrases) >= 8:
            break
    return phrases


def clean_query_topic(topic: str) -> str:
    cleaned = strip_markdown(topic)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 +#.-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    words = [word for word in cleaned.split() if word.lower() not in STOPWORDS]
    return " ".join(words[:8])


def strip_markdown(text: str) -> str:
    cleaned = re.sub(r"[*_`>#\[\]():|]", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_useful_topic(topic: str) -> bool:
    cleaned = clean_youtube_text(topic)
    if len(cleaned) < 4 or len(cleaned) > 140:
        return False
    words = [word for word in re.findall(r"[A-Za-z0-9+#.-]+", cleaned) if word.lower() not in STOPWORDS]
    return bool(words)


def clean_youtube_text(value) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = (
        value.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def choose_thumbnail_url(thumbnails: dict) -> str | None:
    for key in ("high", "medium", "default"):
        thumbnail = thumbnails.get(key)
        if isinstance(thumbnail, dict) and isinstance(thumbnail.get("url"), str):
            return thumbnail["url"]
    return None
