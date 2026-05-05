import httpx

from app.core.config import Settings
from app.services.pdf_text import ExtractedPdf
from app.services.youtube_videos import (
    append_recommended_videos_to_notes,
    build_video_search_queries,
    parse_youtube_search_item,
    search_youtube_learning_videos,
)


def source() -> ExtractedPdf:
    return ExtractedPdf(
        filename="ml-article.pdf",
        text="Gradient Descent for Machine Learning\n\nGradient descent updates model weights.",
        page_count=1,
        extracted_characters=79,
        truncated=False,
    )


def test_build_video_search_queries_prefers_learner_goal_and_note_headings() -> None:
    queries = build_video_search_queries(
        source=source(),
        notes_markdown="# Short Overview\n\n## Gradient Descent\n\n## Learning Rate",
        learner_goal="Understand ML optimization",
        max_queries=3,
    )

    assert queries == [
        "Understand ML optimization beginner explanation tutorial",
        "Gradient Descent beginner explanation tutorial",
        "Learning Rate beginner explanation tutorial",
    ]


def test_parse_youtube_search_item_returns_embed_ready_video() -> None:
    video = parse_youtube_search_item(
        item={
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "Gradient Descent &amp; Learning Rate",
                "channelTitle": "ML Teacher",
                "thumbnails": {"high": {"url": "https://img.example/high.jpg"}},
            },
        },
        search_query="gradient descent beginner explanation tutorial",
    )

    assert video is not None
    assert video.video_id == "abc123"
    assert video.title == "Gradient Descent & Learning Rate"
    assert video.url == "https://www.youtube.com/watch?v=abc123"
    assert video.embed_url == "https://www.youtube.com/embed/abc123"
    assert video.thumbnail_url == "https://img.example/high.jpg"


def test_search_youtube_learning_videos_returns_empty_without_api_key() -> None:
    videos = search_youtube_learning_videos(
        settings=Settings(YOUTUBE_API_KEY=None),
        source=source(),
        notes_markdown="## Gradient Descent",
    )

    assert videos == []


def test_append_recommended_videos_to_notes_adds_visible_links() -> None:
    video = parse_youtube_search_item(
        item={
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "Gradient Descent Explained",
                "channelTitle": "ML Teacher",
                "thumbnails": {},
            },
        },
        search_query="gradient descent beginner explanation tutorial",
    )

    markdown = append_recommended_videos_to_notes("## Notes\n\nLearn the idea.", [video])

    assert "## Recommended Videos" in markdown
    assert "[Gradient Descent Explained](https://www.youtube.com/watch?v=abc123)" in markdown
    assert "Embed URL: https://www.youtube.com/embed/abc123" in markdown


def test_search_youtube_learning_videos_uses_youtube_api_payload() -> None:
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": {"videoId": "abc123"},
                        "snippet": {
                            "title": "Gradient Descent Explained",
                            "channelTitle": "ML Teacher",
                            "thumbnails": {"medium": {"url": "https://img.example/medium.jpg"}},
                        },
                    }
                ]
            },
            request=request,
        )

    videos = search_youtube_learning_videos(
        settings=Settings(YOUTUBE_API_KEY="test-key", YOUTUBE_MAX_VIDEOS=1),
        source=source(),
        notes_markdown="## Gradient Descent",
        http_get=fake_get,
    )

    assert len(videos) == 1
    assert videos[0].video_id == "abc123"
    assert calls[0][1]["key"] == "test-key"
    assert calls[0][1]["type"] == "video"
    assert calls[0][1]["videoEmbeddable"] == "true"
