import httpx
import pytest

from app.services.article_text import ArticleExtractionError, extract_article_text


def test_extract_article_text_reads_article_body(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html>
      <head><title>Test Article</title></head>
      <body>
        <nav>Ignore this</nav>
        <article>
          <h1>Test Article</h1>
          <p>This is a useful explanation about attention and learning.</p>
          <p>It includes enough text for the extractor to accept the article.</p>
          <p>Students can use it to prepare focused study notes quickly.</p>
        </article>
      </body>
    </html>
    """

    def fake_get(*args, **kwargs):
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://example.com/article"),
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    article = extract_article_text("https://example.com/article", max_chars=1000)

    assert article.title == "Test Article"
    assert "attention and learning" in article.text
    assert "Ignore this" not in article.text
    assert article.truncated is False


def test_extract_article_text_rejects_youtube_urls() -> None:
    with pytest.raises(ArticleExtractionError):
        extract_article_text("https://www.youtube.com/watch?v=abc123", max_chars=1000)
