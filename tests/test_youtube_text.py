import pytest

from app.services.youtube_text import YouTubeExtractionError, extract_youtube_video_id


@pytest.mark.parametrize(
    ("url", "video_id"),
    [
        ("https://www.youtube.com/watch?v=abc123", "abc123"),
        ("https://youtu.be/abc123", "abc123"),
        ("https://www.youtube.com/shorts/abc123", "abc123"),
        ("https://www.youtube.com/embed/abc123", "abc123"),
    ],
)
def test_extract_youtube_video_id(url: str, video_id: str) -> None:
    assert extract_youtube_video_id(url) == video_id


def test_extract_youtube_video_id_rejects_non_youtube_url() -> None:
    with pytest.raises(YouTubeExtractionError):
        extract_youtube_video_id("https://example.com/video")
