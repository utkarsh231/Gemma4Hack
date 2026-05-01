from google import genai

from app.core.config import Settings
from app.schemas.notes import DetailLevel, NotesResponse, SourceStats
from app.services.pdf_text import ExtractedPdf
from app.services.youtube_text import extract_youtube_transcript


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

        return build_notes_response(notes_markdown=text, source=extracted_pdf)

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

        return build_notes_response(notes_markdown=text, source=source), source

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
