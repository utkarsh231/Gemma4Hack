# Gemma4Hack Backend

Backend-only first slice: upload a PDF and generate ADHD-friendly study notes using hosted Gemma through the Gemini API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `GEMINI_API_KEY` and `PINECONE_API_KEY` in `.env`.
Set `YOUTUBE_API_KEY` if you want generated notes to include recommended explainer videos.
Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and backend-only `SUPABASE_SERVICE_ROLE_KEY`
to verify auth tokens and persist XP rewards to Supabase.

Google's current Gemma-on-Gemini documentation lists hosted Gemma 4 models through the Gemini API. The backend defaults to `gemma-4-26b-a4b-it`; use `gemma-4-31b-it` when you want the larger model.

For the local Vite frontend, keep:

```env
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

RAG chat uses Pinecone integrated embeddings. By default the app creates/uses a dense index named
`gemma4hack-study-chunks` with `llama-text-embed-v2` in `aws/us-east-1`.
The app writes each chunk to both `chunk_text` and `text` fields so it can work with Pinecone indexes
configured with either field mapping.
Follow-up chat uses app-level hybrid retrieval: fast local keyword matching over stored PDF chunks plus
Pinecone semantic search, with a timeout fallback to keyword-only context if semantic inference is slow.

```env
PINECONE_INDEX_NAME=gemma4hack-study-chunks
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_EMBEDDING_MODEL=llama-text-embed-v2
RAG_TOP_K=5
RAG_KEYWORD_TOP_K=4
RAG_SEMANTIC_TIMEOUT_SECONDS=8
```

Recommended videos use the YouTube Data API. When `YOUTUBE_API_KEY` is configured, note generation extracts a few learning topics from the generated notes/source and returns embed-ready video metadata in `recommended_videos`. If the key is missing or YouTube search fails, note generation still succeeds with an empty video list.

```env
YOUTUBE_API_KEY=
YOUTUBE_MAX_VIDEOS=4
```

## Run

```bash
uvicorn app.main:app --reload
```

## API

Generate notes directly:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/notes/from-pdf" \
  -F "file=@/path/to/notes.pdf" \
  -F "learner_goal=Prepare for tomorrow's quiz" \
  -F "detail_level=standard"
```

The response returns Markdown notes plus source metadata:

```json
{
  "notes_markdown": "### **Topic**\n\n**Overview:**\n...",
  "source_stats": {
    "filename": "notes.pdf",
    "page_count": 10,
    "extracted_characters": 25000,
    "truncated": false
  },
  "recommended_videos": [
    {
      "video_id": "VIDEO_ID",
      "title": "Topic explained",
      "channel_title": "Education Channel",
      "url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "embed_url": "https://www.youtube.com/embed/VIDEO_ID",
      "thumbnail_url": "https://...",
      "search_query": "topic beginner explanation tutorial"
    }
  ]
}
```

Create a chat session from a PDF:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/sessions/from-pdf" \
  -F "file=@/path/to/notes.pdf" \
  -F "learner_goal=Prepare for tomorrow's quiz" \
  -F "detail_level=standard"
```

Create a chat session from a public YouTube video:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/sessions/from-youtube" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "learner_goal": "Prepare for tomorrow'\''s quiz",
    "detail_level": "standard"
  }'
```

Create a chat session from an online article:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/sessions/from-link" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "learner_goal": "Prepare for tomorrow'\''s quiz",
    "detail_level": "standard"
  }'
```

Ask a follow-up question:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/sessions/{session_id}/messages" \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the 3 most important ideas?"}'
```

Complete a session and award XP:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/sessions/{session_id}/complete" \
  -H "Content-Type: application/json" \
  -d '{"actual_duration_seconds":1500}'
```

The completion response includes `xp_earned`, `xp_breakdown`, `xp_summary`, and the updated `chat_session`.
Use `xp_earned` for the session-complete UI, then show `xp_summary.total_xp`, `xp_summary.current_level`,
`xp_summary.current_tier_display_name`, `xp_summary.completed_tracks`, and `xp_summary.total_focus_seconds`
in the learner profile/header.
Include the user's Supabase access token as `Authorization: Bearer <token>` to persist XP to Supabase.
Without a bearer token, local in-memory XP is returned for development.

Fetch the current XP summary:

```bash
curl "http://127.0.0.1:8000/api/v1/chat/xp"
```

Chat sessions are stored in memory, so sessions reset when the server restarts. PDF chunks are indexed in Pinecone under the chat session ID namespace and retrieved for each follow-up question.

## Test

```bash
pytest
```
