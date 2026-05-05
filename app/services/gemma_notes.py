import json
import logging
import re

from google import genai

from app.core.config import Settings
from app.schemas.chat import DiagnosticQuizOption, DiagnosticQuizQuestion, SourceSection
from app.schemas.notes import DetailLevel, NotesResponse, SourceStats
from app.services.article_text import extract_article_text
from app.services.pdf_text import ExtractedPdf
from app.services.youtube_videos import (
    YouTubeVideoSearchError,
    append_recommended_videos_to_notes,
    search_youtube_learning_videos,
)
from app.services.youtube_text import extract_youtube_transcript

logger = logging.getLogger(__name__)


class NotesGenerationError(Exception):
    """Raised when Gemma cannot generate a valid notes payload."""


class GemmaNotesService:
    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def generate_notes(
        self,
        extracted_pdf: ExtractedPdf,
        learner_goal: str | None,
        detail_level: DetailLevel,
    ) -> NotesResponse:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)
        prompt = build_notes_prompt(
            extracted_pdf=extracted_pdf,
            learner_goal=learner_goal,
            detail_level=detail_level,
        )

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=prompt,
            )
        except Exception as exc:
            raise NotesGenerationError("Gemma request failed.") from exc

        text = getattr(response, "text", None)
        if not text:
            raise NotesGenerationError("Gemma returned an empty response.")

        notes = build_notes_response(notes_markdown=text, source=extracted_pdf)
        return self.add_video_recommendations(source=extracted_pdf, notes=notes, learner_goal=learner_goal)

    def generate_notes_from_article(
        self,
        *,
        url: str,
        learner_goal: str | None,
        detail_level: DetailLevel,
    ) -> tuple[NotesResponse, ExtractedPdf]:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)
        article = extract_article_text(url, max_chars=self.settings.max_extracted_chars)
        source = ExtractedPdf(
            filename=article.url,
            text=f"{article.title}\n\n{article.text}",
            page_count=1,
            extracted_characters=article.extracted_characters,
            truncated=article.truncated,
        )
        prompt = build_article_notes_prompt(
            source=source,
            title=article.title,
            learner_goal=learner_goal,
            detail_level=detail_level,
        )

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=prompt,
            )
        except Exception as exc:
            raise NotesGenerationError(f"Gemma article request failed with model {self.settings.gemma_model}: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise NotesGenerationError("Gemma returned an empty response.")

        notes = build_notes_response(notes_markdown=text, source=source)
        return self.add_video_recommendations(source=source, notes=notes, learner_goal=learner_goal), source

    def generate_notes_from_youtube(
        self,
        *,
        youtube_url: str,
        learner_goal: str | None,
        detail_level: DetailLevel,
    ) -> tuple[NotesResponse, ExtractedPdf]:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)
        transcript = extract_youtube_transcript(youtube_url, max_chars=self.settings.max_extracted_chars)
        source = ExtractedPdf(
            filename=youtube_url,
            text=transcript.text,
            page_count=1,
            extracted_characters=transcript.extracted_characters,
            truncated=transcript.truncated,
        )
        prompt = build_youtube_notes_prompt(
            source=source,
            learner_goal=learner_goal,
            detail_level=detail_level,
        )

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=prompt,
            )
        except Exception as exc:
            raise NotesGenerationError(f"Gemma YouTube transcript request failed with model {self.settings.gemma_model}: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise NotesGenerationError("Gemma returned an empty response.")

        notes = build_notes_response(notes_markdown=text, source=source)
        return self.add_video_recommendations(source=source, notes=notes, learner_goal=learner_goal), source

    def answer_question(
        self,
        *,
        source: ExtractedPdf,
        notes_markdown: str,
        conversation_markdown: str,
        retrieved_context: str,
        question: str,
    ) -> str:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)
        prompt = build_chat_prompt(
            source=source,
            notes_markdown=notes_markdown,
            conversation_markdown=conversation_markdown,
            retrieved_context=retrieved_context,
            question=question,
        )

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=prompt,
            )
        except Exception as exc:
            raise NotesGenerationError("Gemma request failed.") from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise NotesGenerationError("Gemma returned an empty response.")
        return text.strip()

    def generate_source_sections(self, *, source: ExtractedPdf, learner_goal: str | None) -> list[SourceSection]:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=build_source_sections_prompt(source=source, learner_goal=learner_goal, max_source_chars=14_000),
            )
        except Exception:
            try:
                response = client.models.generate_content(
                    model=self.settings.gemma_model,
                    contents=build_source_sections_prompt(source=source, learner_goal=learner_goal, max_source_chars=6_000),
                )
            except Exception:
                return fallback_source_sections(source)

        text = getattr(response, "text", None)
        if not text or not text.strip():
            return fallback_source_sections(source)
        try:
            return parse_source_sections(text)
        except NotesGenerationError:
            return fallback_source_sections(source)

    def generate_diagnostic_quiz(
        self,
        *,
        sections: list[SourceSection],
        learner_goal: str | None,
    ) -> list[DiagnosticQuizQuestion]:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=build_diagnostic_quiz_prompt(sections=sections, learner_goal=learner_goal),
            )
        except Exception:
            try:
                response = client.models.generate_content(
                    model=self.settings.gemma_model,
                    contents=build_compact_diagnostic_quiz_prompt(sections=sections, learner_goal=learner_goal),
                )
            except Exception:
                return self.generate_diagnostic_quiz_by_section(
                    client=client,
                    sections=sections,
                    learner_goal=learner_goal,
                )

        text = getattr(response, "text", None)
        if not text or not text.strip():
            return self.generate_diagnostic_quiz_by_section(
                client=client,
                sections=sections,
                learner_goal=learner_goal,
            )
        try:
            return parse_diagnostic_quiz(text)
        except NotesGenerationError:
            return self.generate_diagnostic_quiz_by_section(
                client=client,
                sections=sections,
                learner_goal=learner_goal,
            )

    def generate_diagnostic_quiz_by_section(
        self,
        *,
        client,
        sections: list[SourceSection],
        learner_goal: str | None,
    ) -> list[DiagnosticQuizQuestion]:
        questions: list[DiagnosticQuizQuestion] = []

        for index, section in enumerate(sections):
            try:
                response = client.models.generate_content(
                    model=self.settings.gemma_model,
                    contents=build_single_section_quiz_prompt(section=section, learner_goal=learner_goal, index=index),
                )
                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise NotesGenerationError("Gemma returned an empty single-section quiz response.")
                parsed = parse_diagnostic_quiz(text)
                question = parsed[0].model_copy(
                    update={
                        "id": f"q{index + 1}",
                        "unit_title": section.title,
                        "source_excerpt": parsed[0].source_excerpt or section.source_excerpt,
                    }
                )
            except Exception:
                question = fallback_quiz_question_from_section(section=section, index=index, all_sections=sections)
            questions.append(question)

        return questions

    def generate_focused_notes(
        self,
        *,
        source: ExtractedPdf,
        learner_goal: str | None,
        detail_level: DetailLevel,
        quiz_markdown: str,
    ) -> NotesResponse:
        if not self.settings.gemini_api_key:
            raise NotesGenerationError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self.settings.gemini_api_key)

        try:
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=build_focused_notes_prompt(
                    source=source,
                    learner_goal=learner_goal,
                    detail_level=detail_level,
                    quiz_markdown=quiz_markdown,
                    max_source_chars=16_000,
                ),
            )
        except Exception:
            try:
                response = client.models.generate_content(
                    model=self.settings.gemma_model,
                    contents=build_focused_notes_prompt(
                        source=source,
                        learner_goal=learner_goal,
                        detail_level=detail_level,
                        quiz_markdown=quiz_markdown,
                        max_source_chars=8_000,
                    ),
                )
            except Exception:
                notes = fallback_focused_notes(source=source, quiz_markdown=quiz_markdown)
                return self.add_video_recommendations(source=source, notes=notes, learner_goal=learner_goal)

        text = getattr(response, "text", None)
        if not text or not text.strip():
            notes = fallback_focused_notes(source=source, quiz_markdown=quiz_markdown)
            return self.add_video_recommendations(source=source, notes=notes, learner_goal=learner_goal)
        notes = build_notes_response(notes_markdown=text, source=source)
        return self.add_video_recommendations(source=source, notes=notes, learner_goal=learner_goal)

    def add_video_recommendations(
        self,
        *,
        source: ExtractedPdf,
        notes: NotesResponse,
        learner_goal: str | None,
    ) -> NotesResponse:
        try:
            videos = search_youtube_learning_videos(
                settings=self.settings,
                source=source,
                notes_markdown=notes.notes_markdown,
                learner_goal=learner_goal,
            )
        except YouTubeVideoSearchError:
            logger.warning("youtube_video_recommendations_failed", exc_info=True)
            videos = []
        notes_markdown = append_recommended_videos_to_notes(notes.notes_markdown, videos)
        return notes.model_copy(update={"notes_markdown": notes_markdown, "recommended_videos": videos})


def build_notes_prompt(
    *,
    extracted_pdf: ExtractedPdf,
    learner_goal: str | None,
    detail_level: DetailLevel,
) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand and remember the PDF."
    return f"""
Generate concise and engaging notes from the provided document, specifically tailored for individuals with ADHD. The goal is to present information in a way that enhances comprehension and retention for users who may experience challenges with attention span.

Learner goal:
{goal}

Detail level:
{detail_level.value}

# Steps

1. **Read and Understand the Document:** Thoroughly analyze the provided document to grasp its core concepts, arguments, and details.
2. **Identify Core Information:** Extract the most crucial information that needs to be conveyed.
3. **Adapt to ADHD Needs:** Reframe and present the extracted information using the following strategies:
   * **Short Overview:** Begin with a brief, high-level summary of the document's main topic.
   * **Key Points:** List the most important takeaways in a clear, scannable format, such as bullet points.
   * **Small Focus Blocks:** Break down complex topics into smaller, digestible sections. Each section should focus on a single idea.
   * **Why This Matters:** Clearly articulate the relevance and importance of the information to the user's life or understanding.
   * **Bite-Sized Notes:** Condense information into short, easily processed sentences or phrases. Avoid long, dense paragraphs.
   * **Quick Self-Check Questions:** Include simple questions after sections to help users gauge their understanding.
   * **Memory Hooks:** Create memorable phrases, analogies, or simple associations to aid recall.
   * **Action Steps:** If applicable, provide clear, actionable steps that the user can take based on the information.
   * **Attention Breakpoints:** Strategically place visual or textual cues, such as a short break sentence, to signal a natural pause point and help reset attention.
4. **Structure the Output:** Organize the notes logically, following the outlined strategies. Ensure a clear flow from overview to detailed points and action steps.
5. **Refine Language:** Use simple, direct language. Avoid jargon where possible or explain it clearly. Maintain an encouraging and supportive tone.

# Output Format

Return only the final notes in Markdown. Do not wrap the output in a code block. Use headings, bullet points, and bold text for emphasis. Ensure that each section is clearly delineated and easy to scan.

# Examples

**Document Input:** A detailed explanation of photosynthesis.

**Output Notes:**

### **Photosynthesis: How Plants Make Food**

**Overview:**
Photosynthesis is the amazing process plants use to turn sunlight, water, and air into their own food (sugar) and oxygen.

**Key Points:**
*   Plants need sunlight, water, and carbon dioxide.
*   Chlorophyll (the green stuff in leaves) captures sunlight.
*   The process creates sugar (food for the plant) and oxygen (which we breathe!).
*   It happens mainly in the leaves.

**Focus Block 1: The Ingredients**
*   **Sunlight:** The energy source. Like a solar panel for plants! ☀️
*   **Water:** Absorbed through the roots. 💧
*   **Carbon Dioxide:** Taken from the air through tiny holes in leaves. 🌬️

**Why This Matters:**
Without photosynthesis, plants couldn't survive, and we wouldn't have the oxygen we need to live! It's the foundation of most life on Earth.

**Bite-Sized Notes:**
*   Sun + Water + CO2 → Sugar + Oxygen
*   Green leaves are like tiny food factories.

**Self-Check:**
Can you name the three main things plants need for photosynthesis? (Sunlight, Water, Carbon Dioxide)

**Memory Hook:**
Think "SUN-drinker" for plants using sunlight.

**Action Steps:**
*   Observe plants around you and imagine them doing photosynthesis.
*   Water your plants regularly!

**Attention Breakpoint:**
--- Take a deep breath! ---

**Focus Block 2: The Process in Leaves**
*   Inside leaves, chlorophyll traps sunlight.
*   This energy is used to combine water and carbon dioxide.
*   This combination creates glucose (sugar) and releases oxygen.

**Why This Matters:**
This is where the "magic" happens, converting simple ingredients into usable energy and essential oxygen.

**Bite-Sized Notes:**
*   Chlorophyll = Sun Catcher
*   Sugar = Plant Food
*   Oxygen = Our Air

**Self-Check:**
What is the plant's food called? (Glucose/Sugar) What gas is released? (Oxygen)

**Memory Hook:**
"Photo" means light, "synthesis" means putting together. Photosynthesis = Putting things together with light.

**Action Steps:**
*   Notice the green color of leaves – that's the chlorophyll working!

**Attention Breakpoint:**
--- Quick Stretch! ---

**(Continuing with other sections as needed, e.g., The Products, Importance for Ecosystems)**

# Notes

*   Emphasize visual cues and clear, bolded headings for each strategy.
*   Keep sentences short and to the point.
*   Use emojis judiciously to add visual interest and break up text.
*   Ensure questions are simple and directly related to the preceding content.
*   Use plain Markdown only. Do not use LaTeX syntax. Use "→" for arrows instead of "$\\rightarrow$".
*   Put a space after every bullet marker. Use "* item" instead of "*item".
*   For attention breakpoints, write plain text such as "--- Quick stretch! ---" without italic markers around the sentence.

PDF metadata:
- filename: {extracted_pdf.filename}
- pages: {extracted_pdf.page_count}
- text truncated: {extracted_pdf.truncated}

PDF text:
{extracted_pdf.text}
""".strip()


def build_article_notes_prompt(
    *,
    source: ExtractedPdf,
    title: str,
    learner_goal: str | None,
    detail_level: DetailLevel,
) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand and remember this article."
    return f"""
Create ADHD-friendly study notes from this online article.

Learner goal:
{goal}

Detail level:
{detail_level.value}

Use the article text as the source of truth.

# Output Format

Return only the final notes in Markdown. Do not wrap the output in a code block.

Include these sections:
* Short overview
* Key takeaways
* Focus blocks
* Important definitions
* Why this matters
* Quick self-check questions
* Memory hooks
* Action steps, if applicable

Style requirements:
* Use simple, direct language.
* Keep paragraphs short.
* Prefer bullets and clear headings.
* Do not diagnose, provide medical advice, or claim the learner has ADHD.

Article source:
- title: {title}
- url: {source.filename}
- article truncated: {source.truncated}

Article text:
{source.text}
""".strip()


def build_youtube_notes_prompt(*, source: ExtractedPdf, learner_goal: str | None, detail_level: DetailLevel) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand and remember the YouTube video."
    return f"""
Create ADHD-friendly study notes from this YouTube transcript.

Learner goal:
{goal}

Detail level:
{detail_level.value}

Use the transcript as the source of truth. Keep the provided timestamps in the notes whenever they help the learner find the relevant part of the video.

# Output Format

Return only the final notes in Markdown. Do not wrap the output in a code block.

Include these sections:
* Short overview
* Key takeaways
* Timestamped focus blocks using timestamps like [00:00] or [01:23]
* Important definitions
* Why this matters
* Quick self-check questions
* Memory hooks
* Action steps, if applicable

Style requirements:
* Use simple, direct language.
* Keep paragraphs short.
* Prefer bullets and clear headings.
* Use timestamps whenever possible.
* If a timestamp is uncertain, say "around" before the timestamp.
* Do not diagnose, provide medical advice, or claim the learner has ADHD.

YouTube source:
- url: {source.filename}
- transcript truncated: {source.truncated}

Timestamped transcript:
{source.text}
""".strip()


def build_chat_prompt(
    *,
    source: ExtractedPdf,
    notes_markdown: str,
    conversation_markdown: str,
    retrieved_context: str,
    question: str,
) -> str:
    return f"""
You are a supportive study assistant helping a learner understand an uploaded document.

Answer the user's question using the retrieved document context and generated notes as your source of truth.

Style requirements:
* Use simple, direct language.
* Keep the answer concise unless the user asks for depth.
* Use Markdown with short headings and bullets when helpful.
* Include a quick self-check question when it helps retention.
* If the answer is not supported by the retrieved context or notes, say that clearly and explain what the available context does say.
* Do not diagnose, provide medical advice, or claim the learner has ADHD.

Document metadata:
- filename: {source.filename}
- pages: {source.page_count}
- text truncated: {source.truncated}

Generated notes:
{notes_markdown}

Recent conversation:
{conversation_markdown}

User question:
{question}

Retrieved document context:
{retrieved_context}
""".strip()


def source_excerpt(source: ExtractedPdf, *, max_chars: int) -> str:
    text = source.text.strip()
    if len(text) <= max_chars:
        return text

    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        f"{text[:head_chars].strip()}\n\n"
        "--- Source excerpt trimmed for this model call. Later material continues below. ---\n\n"
        f"{text[-tail_chars:].strip()}"
    )


def build_source_sections_prompt(*, source: ExtractedPdf, learner_goal: str | None, max_source_chars: int) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand this study material."
    excerpt = source_excerpt(source, max_chars=max_source_chars)
    return f"""
Divide the source material into major learning sections for a diagnostic quiz.

Learner goal:
{goal}

Requirements:
* Create 3 to 6 meaningful sections.
* Ignore citation boilerplate, author lists, page headers, page numbers, URLs, and journal metadata unless they are central to the content.
* Each section must represent a real concept, claim, method, finding, or example from the source.
* For each section, include a concise summary and a clean source excerpt that supports the summary.
* Keep language simple and direct.
* Use only the source material as truth.

Return only valid JSON. Do not wrap it in a code block.

JSON shape:
{{
  "sections": [
    {{
      "id": "s1",
      "title": "short section title",
      "summary": "what this section teaches",
      "source_excerpt": "clean supporting excerpt from the source"
    }}
  ]
}}

Source metadata:
- filename: {source.filename}
- pages: {source.page_count}
- text truncated: {source.truncated}

Source text:
{excerpt}
""".strip()


def build_diagnostic_quiz_prompt(*, sections: list[SourceSection], learner_goal: str | None) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand this study material."
    sections_json = json.dumps([section.model_dump() for section in sections], ensure_ascii=False, indent=2)
    return f"""
Create a diagnostic warm-up quiz from these already-cleaned learning sections.

Learner goal:
{goal}

Requirements:
* Write exactly one multiple-choice question per section.
* Questions must test understanding of the section's concept, not citation metadata or wording recall.
* Each question must have 4 plausible options.
* Only one option may be correct.
* Distractors should be believable misconceptions based on the same section.
* Do not copy raw citation strings, author lists, URLs, page labels, or journal headings into options.
* Keep questions and options simple, natural, and student-facing.
* Use only the provided sections as truth.

Return only valid JSON. Do not wrap it in a code block.

JSON shape:
{{
  "questions": [
    {{
      "id": "q1",
      "unit_title": "section title",
      "question": "question text?",
      "options": [
        {{ "id": "A", "text": "option text" }},
        {{ "id": "B", "text": "option text" }},
        {{ "id": "C", "text": "option text" }},
        {{ "id": "D", "text": "option text" }}
      ],
      "correct_option_id": "A",
      "explanation": "Briefly explain why the correct option is right.",
      "key_takeaway": "the main idea this question checks",
      "study_note": "a short explanation to use later in adaptive notes",
      "source_excerpt": "the relevant clean excerpt from the section"
    }}
  ]
}}

Learning sections:
{sections_json}
""".strip()


def build_compact_diagnostic_quiz_prompt(*, sections: list[SourceSection], learner_goal: str | None) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand this study material."
    compact_sections = [
        {
            "id": section.id,
            "title": section.title,
            "summary": section.summary[:500],
        }
        for section in sections
    ]
    return f"""
Create one clean multiple-choice understanding question per section.

Learner goal: {goal}

Rules:
* Use only these section summaries.
* Ignore citations and metadata.
* Each question has 4 options, exactly one correct answer.
* Options must be natural learner-facing statements, not raw copied text.

Return only valid JSON with this shape:
{{"questions":[{{"id":"q1","unit_title":"...","question":"...?","options":[{{"id":"A","text":"..."}},{{"id":"B","text":"..."}},{{"id":"C","text":"..."}},{{"id":"D","text":"..."}}],"correct_option_id":"A","explanation":"...","key_takeaway":"...","study_note":"...","source_excerpt":"..."}}]}}

Sections:
{json.dumps(compact_sections, ensure_ascii=False)}
""".strip()


def build_single_section_quiz_prompt(*, section: SourceSection, learner_goal: str | None, index: int) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand this study material."
    return f"""
Create exactly one multiple-choice understanding question for this section.

Learner goal: {goal}

Rules:
* Test the concept, not citation details or wording recall.
* Use 4 natural options.
* Exactly one option is correct.
* Return only valid JSON.

JSON shape:
{{"questions":[{{"id":"q{index + 1}","unit_title":"{section.title}","question":"...?","options":[{{"id":"A","text":"..."}},{{"id":"B","text":"..."}},{{"id":"C","text":"..."}},{{"id":"D","text":"..."}}],"correct_option_id":"A","explanation":"...","key_takeaway":"...","study_note":"...","source_excerpt":"..."}}]}}

Section:
{json.dumps(section.model_dump(), ensure_ascii=False)}
""".strip()


def build_focused_notes_prompt(
    *,
    source: ExtractedPdf,
    learner_goal: str | None,
    detail_level: DetailLevel,
    quiz_markdown: str,
    max_source_chars: int,
) -> str:
    goal = learner_goal.strip() if learner_goal and learner_goal.strip() else "Help the learner understand and remember this material."
    excerpt = source_excerpt(source, max_chars=max_source_chars)
    return f"""
Generate ADHD-friendly study notes from the source material, adapted to the learner's diagnostic quiz performance.

Learner goal:
{goal}

Detail level:
{detail_level.value}

Diagnostic quiz performance:
{quiz_markdown}

Adaptation rules:
* Focus more deeply on units where the learner answered incorrectly or reported confidence below 60%.
* For low-confidence units, add a clearer explanation, one concrete example, and a quick self-check.
* For correct high-confidence units, keep the notes shorter and avoid overexplaining.
* Do not shame the learner. Use a supportive tone.
* Do not include the full quiz or answer key in the notes.
* Use the source material as the source of truth.

Output format:
Return only Markdown. Do not wrap the output in a code block.
Include:
* Personalized focus map
* Short overview
* Key takeaways
* Focus blocks, weighted toward weak or low-confidence areas
* Important definitions
* Why this matters
* Quick self-check questions
* Memory hooks only when they clarify the actual concept
* Attention breakpoints

Source metadata:
- filename: {source.filename}
- pages: {source.page_count}
- text truncated: {source.truncated}

Source text:
{excerpt}
""".strip()


def parse_diagnostic_quiz(raw_text: str) -> list[DiagnosticQuizQuestion]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise NotesGenerationError("Gemma returned quiz JSON that could not be parsed.") from exc

    questions_payload = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions_payload, list):
        raise NotesGenerationError("Gemma quiz response did not include a questions list.")

    try:
        questions = [DiagnosticQuizQuestion.model_validate(question) for question in questions_payload]
    except Exception as exc:
        raise NotesGenerationError("Gemma quiz response did not match the expected schema.") from exc

    for question in questions:
        option_ids = {option.id for option in question.options}
        if question.correct_option_id not in option_ids:
            raise NotesGenerationError("Gemma quiz response had a correct option id that was not in the options.")
    return questions


def parse_source_sections(raw_text: str) -> list[SourceSection]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise NotesGenerationError("Gemma returned source section JSON that could not be parsed.") from exc

    sections_payload = payload.get("sections") if isinstance(payload, dict) else None
    if not isinstance(sections_payload, list):
        raise NotesGenerationError("Gemma source section response did not include a sections list.")

    try:
        return [SourceSection.model_validate(section) for section in sections_payload]
    except Exception as exc:
        raise NotesGenerationError("Gemma source section response did not match the expected schema.") from exc


def fallback_source_sections(source: ExtractedPdf) -> list[SourceSection]:
    paragraphs = clean_source_paragraphs(source.text)
    if not paragraphs:
        paragraphs = ["The source material was extracted, but it did not contain enough clean paragraph text for automatic sectioning."]

    target_count = max(1, min(5, len(paragraphs)))
    buckets = [[] for _ in range(target_count)]
    for index, paragraph in enumerate(paragraphs[: target_count * 3]):
        buckets[index % target_count].append(paragraph)

    sections: list[SourceSection] = []
    for index, bucket in enumerate(buckets):
        text = " ".join(bucket).strip()
        if not text:
            continue
        summary = summarize_paragraph_text(text)
        sections.append(
            SourceSection(
                id=f"s{index + 1}",
                title=f"Focus Area {index + 1}",
                summary=summary,
                source_excerpt=text[:1800],
            )
        )

    return sections or [
        SourceSection(
            id="s1",
            title="Focus Area 1",
            summary=paragraphs[0][:1000],
            source_excerpt=paragraphs[0][:1800],
        )
    ]


def clean_source_paragraphs(text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n|\n(?=\S)", text)
    paragraphs: list[str] = []

    for block in raw_blocks:
        cleaned = re.sub(r"\s+", " ", block).strip()
        if not is_useful_source_paragraph(cleaned):
            continue
        paragraphs.append(cleaned)
        if len(paragraphs) >= 18:
            break

    if paragraphs:
        return paragraphs

    return [
        sentence
        for sentence in extract_quiz_sentences(text)
        if is_useful_source_paragraph(sentence)
    ][:12]


def is_useful_source_paragraph(text: str) -> bool:
    if len(text) < 80:
        return False
    lower = text.lower()
    metadata_markers = [
        "doi.org",
        "http://",
        "https://",
        "arxiv",
        "journal",
        "volume",
        "copyright",
        "license",
        "correspondence",
        "received:",
        "accepted:",
        "published:",
    ]
    if any(marker in lower for marker in metadata_markers):
        return False
    if re.search(r"\b\d+\s*\|\s*\w+", text):
        return False
    alpha_ratio = sum(char.isalpha() for char in text) / max(1, len(text))
    return alpha_ratio > 0.55


def summarize_paragraph_text(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if 50 <= len(sentence.strip()) <= 260
    ]
    summary = " ".join(sentences[:2]).strip() or text[:500]
    return summary[:1000]


def fallback_quiz_question_from_section(
    *,
    section: SourceSection,
    index: int,
    all_sections: list[SourceSection],
) -> DiagnosticQuizQuestion:
    other_summaries = [
        other.summary
        for other in all_sections
        if other.id != section.id and other.summary.strip()
    ]
    generic_distractors = [
        f"It mainly says {section.title} is only a citation detail and not a concept.",
        f"It argues that {section.title} has no connection to the source's main evidence.",
        f"It says the learner only needs to memorize the title {section.title}.",
    ]
    option_texts = [section.summary, *other_summaries, *generic_distractors]
    deduped_options: list[str] = []
    for option in option_texts:
        cleaned = re.sub(r"\s+", " ", option).strip()
        if not cleaned or cleaned.lower() in {item.lower() for item in deduped_options}:
            continue
        deduped_options.append(cleaned[:500])
        if len(deduped_options) == 4:
            break

    while len(deduped_options) < 4:
        deduped_options.append(generic_distractors[len(deduped_options) % len(generic_distractors)])

    option_ids = ["A", "B", "C", "D"]
    return DiagnosticQuizQuestion(
        id=f"q{index + 1}",
        unit_title=section.title,
        question=f"Which statement best captures the main idea of {section.title}?",
        options=[
            DiagnosticQuizOption(id=option_ids[option_index], text=option)
            for option_index, option in enumerate(deduped_options)
        ],
        correct_option_id="A",
        explanation="The correct option matches the cleaned section summary.",
        key_takeaway=section.summary,
        study_note=section.summary,
        source_excerpt=section.source_excerpt,
    )


def extract_quiz_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if 60 <= len(sentence.strip()) <= 260 and not sentence.strip().endswith("?")
    ]
    unique: list[str] = []
    for sentence in candidates:
        normalized = sentence.lower()
        if normalized in {item.lower() for item in unique}:
            continue
        unique.append(sentence)
        if len(unique) >= 8:
            break
    return unique


def generic_quiz_distractors() -> list[str]:
    return [
        "The source mainly says the topic is unrelated to the examples, so the examples can be ignored.",
        "The source argues that the key result is random and does not need evidence.",
        "The source focuses only on memorizing terms, not understanding relationships between ideas.",
    ]


def fallback_focused_notes(*, source: ExtractedPdf, quiz_markdown: str) -> NotesResponse:
    sentences = extract_quiz_sentences(source.text)
    focus_areas = extract_low_confidence_focus_areas(quiz_markdown)
    overview = sentences[0] if sentences else "The source material has been extracted and is ready for focused study."
    key_points = sentences[1:5] or [overview]

    focus_lines = "\n".join(
        f"* **{area}:** Spend extra time here because the quiz result or confidence score showed this needs support."
        for area in focus_areas[:4]
    )
    if not focus_lines:
        focus_lines = "* Your quiz responses looked steady overall. Use the focus blocks below as a quick review path."

    key_point_lines = "\n".join(f"* {point}" for point in key_points)
    focus_block_lines = "\n\n".join(
        [
            f"### Focus Block {index + 1}\n\n"
            f"* {sentence}\n"
            "* Pause after reading this and explain it back in one sentence.\n"
            "* **Quick self-check:** What is the main relationship or idea in this block?"
            for index, sentence in enumerate(key_points[:3])
        ]
    )

    notes_markdown = f"""
## Personalized Focus Map

{focus_lines}

## Short Overview

{overview}

## Key Takeaways

{key_point_lines}

{focus_block_lines}

## Why This Matters

These notes were generated from a fallback path because the live model provider returned an internal error during adaptive note generation. The quiz results were still used to choose where to place extra attention.

## Memory Hook

Think: question -> confidence -> focus. Low confidence means slow down and rebuild the idea from the source.

## Attention Breakpoint

--- Take one breath, then review the first focus block again. ---
""".strip()

    return build_notes_response(notes_markdown=notes_markdown, source=source)


def extract_low_confidence_focus_areas(quiz_markdown: str) -> list[str]:
    blocks = re.split(r"\n(?=##\s+)", quiz_markdown.strip())
    focus_areas: list[str] = []

    for block in blocks:
        title_match = re.search(r"^##\s+(.+)$", block, flags=re.MULTILINE)
        needs_focus = re.search(r"Needs extra focus:\s*yes", block, flags=re.IGNORECASE)
        if title_match and needs_focus:
            focus_areas.append(title_match.group(1).strip())

    return focus_areas


def build_notes_response(*, notes_markdown: str, source: ExtractedPdf) -> NotesResponse:
    cleaned_notes = notes_markdown.strip()
    if not cleaned_notes:
        raise NotesGenerationError("Gemma returned empty notes.")

    return NotesResponse(
        notes_markdown=cleaned_notes,
        source_stats=SourceStats(
            filename=source.filename,
            page_count=source.page_count,
            extracted_characters=source.extracted_characters,
            truncated=source.truncated,
        ),
    )


def fallback_notes(source: ExtractedPdf) -> NotesResponse:
    return NotesResponse(
        notes_markdown=(
            "### **Study Notes**\n\n"
            "**Overview:**\n"
            "The source text was extracted successfully, but live note generation was not called.\n\n"
            "**Key Points:**\n"
            "* Upload handling and PDF extraction are ready.\n"
            "* Configure `GEMINI_API_KEY` and call the notes endpoint with a real PDF.\n\n"
            "**Attention Breakpoint:**\n"
            "--- Pause here before testing the live model. ---"
        ),
        source_stats=SourceStats(
            filename=source.filename,
            page_count=source.page_count,
            extracted_characters=source.extracted_characters,
            truncated=source.truncated,
        ),
    )
