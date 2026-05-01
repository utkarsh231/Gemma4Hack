from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


class YouTubeExtractionError(Exception):
    """Raised when a YouTube URL cannot be converted into usable text."""


@dataclass(frozen=True)
class ExtractedYouTubeTranscript:
    video_id: str
    text: str
    extracted_characters: int
    truncated: bool


def extract_youtube_transcript(url: str, *, max_chars: int) -> ExtractedYouTubeTranscript:
    video_id = extract_youtube_video_id(url)

    try:
        rows = _fetch_transcript_rows(video_id)
    except ImportError as exc:
        raise YouTubeExtractionError(
            "The youtube-transcript-api package is not installed. Run `pip install -e .` after updating dependencies."
        ) from exc
    except Exception as exc:
        raise YouTubeExtractionError(
            "Could not read captions for this YouTube video. Use a public video with captions enabled."
        ) from exc

    parts: list[str] = []
    truncated = False
    total_chars = 0
    for row in rows:
        text = " ".join(row["text"].split())
        if not text:
            continue

        next_part = f"[{_format_timestamp(row['start'])}] {text}"
        next_len = len(next_part) + (1 if parts else 0)
        if total_chars + next_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                parts.append(next_part[:remaining])
            truncated = True
            break

        parts.append(next_part)
        total_chars += next_len

    transcript_text = "\n".join(parts).strip()
    if len(transcript_text) < 100:
        raise YouTubeExtractionError("The YouTube transcript did not contain enough usable text.")

    return ExtractedYouTubeTranscript(
        video_id=video_id,
        text=transcript_text,
        extracted_characters=len(transcript_text),
        truncated=truncated,
    )


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and parsed.path.startswith("/shorts/"):
        video_id = parsed.path.removeprefix("/shorts/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and parsed.path.startswith("/embed/"):
        video_id = parsed.path.removeprefix("/embed/").split("/")[0]
    else:
        video_id = ""

    if not video_id:
        raise YouTubeExtractionError("Only YouTube video URLs are supported.")
    return video_id


def _fetch_transcript_rows(video_id: str) -> list[dict[str, float | str]]:
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        return [_row_from_snippet(snippet) for snippet in transcript]
    except AttributeError:
        rows = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return [_row_from_mapping(row) for row in rows]


def _row_from_snippet(snippet) -> dict[str, float | str]:
    return {
        "text": str(getattr(snippet, "text")),
        "start": float(getattr(snippet, "start")),
        "duration": float(getattr(snippet, "duration", 0.0)),
    }


def _row_from_mapping(row: dict) -> dict[str, float | str]:
    return {
        "text": str(row.get("text", "")),
        "start": float(row.get("start", 0.0)),
        "duration": float(row.get("duration", 0.0)),
    }


def _format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
