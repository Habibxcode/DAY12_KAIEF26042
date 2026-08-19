"""
rag.py
------
Retrieval-Augmented Generation (RAG) inference layer.

Responsible for:
  - Initialising the Google Gemini generative model.
  - Maintaining optional multi-turn conversation history.
  - Constructing a strict, grounded RAG prompt that prevents hallucination.
  - Calling the Gemini API and returning a clean text response.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "gemini-2.5-flash"
FALLBACK_MODEL: str = "gemini-1.5-flash"

# Maximum characters of context to send per query (avoids token overflow).
MAX_CONTEXT_CHARS: int = 6000

RAG_SYSTEM_PROMPT: str = """You are a helpful and precise FAQ assistant for ACME Corporation.

Your role is to answer employee questions **strictly** based on the provided
CONTEXT extracted from the official company knowledge base.

RULES YOU MUST FOLLOW:
1. Answer ONLY from the information present in the CONTEXT sections below.
2. Do NOT use any external knowledge, assumptions, or prior training data.
3. If the CONTEXT does not contain sufficient information to answer the question,
   respond with exactly:
   "I am sorry, but that information is not available in the knowledge base."
4. Keep your answers clear, concise, and professional.
5. When relevant, cite the section or policy name from the context.
6. Do not fabricate numbers, dates, percentages, or policy details.
7. If the question is a follow-up, use the conversation history for pronoun
   resolution but still ground your answer in the CONTEXT.
"""


# ---------------------------------------------------------------------------
# Conversation History
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single exchange in the conversation history."""
    role: str   # "user" or "assistant"
    text: str


@dataclass
class ConversationHistory:
    """
    Stores the last N turns of conversation for multi-turn context.

    Attributes:
        max_turns: Maximum number of user-assistant turn pairs to retain.
        turns:     Ordered list of Turn objects.
    """
    max_turns: int = 5
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, text: str) -> None:
        """Append a new turn and evict the oldest pair if over capacity."""
        self.turns.append(Turn(role=role, text=text))
        # Keep at most max_turns * 2 entries (user + assistant per turn).
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def format(self) -> str:
        """
        Format history as a compact dialogue block for prompt injection.

        Returns:
            A multi-line string of prior Q&A exchanges, or empty string
            if no history exists.
        """
        if not self.turns:
            return ""
        lines = ["--- PREVIOUS CONVERSATION ---"]
        for turn in self.turns:
            prefix = "Employee" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.text}")
        lines.append("--- END OF HISTORY ---")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the conversation history."""
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns) // 2


# ---------------------------------------------------------------------------
# Gemini Model Initialisation
# ---------------------------------------------------------------------------

def _create_model(model_name: str) -> genai.GenerativeModel:
    """Instantiate a Gemini GenerativeModel with the RAG system prompt."""
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=RAG_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=800,
        ),
    )


def initialise_gemini(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure the Google Generative AI SDK and return the Gemini model.

    Tries DEFAULT_MODEL first, falls back to FALLBACK_MODEL.

    Args:
        api_key: Optional API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        An initialised GenerativeModel ready for inference.

    Raises:
        EnvironmentError: If no API key is available.
        RuntimeError:     If both models fail to initialise.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Create a .env file with GEMINI_API_KEY=<your_key>."
        )

    genai.configure(api_key=key)

    for model_name in (DEFAULT_MODEL, FALLBACK_MODEL):
        try:
            model = _create_model(model_name)
            logger.info("Gemini model initialised: %s", model_name)
            return model
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not load model '%s': %s", model_name, exc)

    raise RuntimeError(
        f"Failed to initialise '{DEFAULT_MODEL}' and '{FALLBACK_MODEL}'. "
        "Check your API key and network connection."
    )


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _truncate_context(chunks: list[str], max_chars: int = MAX_CONTEXT_CHARS) -> list[str]:
    """
    Truncate the list of context chunks so the total character count stays
    within `max_chars`, preserving the highest-ranked chunks first.

    Args:
        chunks:    Ordered list of retrieved chunk strings (best first).
        max_chars: Maximum total characters to include.

    Returns:
        A subset of chunks within the character budget.
    """
    selected: list[str] = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)
    if len(selected) < len(chunks):
        logger.debug(
            "Context truncated from %d to %d chunks (%d chars budget).",
            len(chunks), len(selected), max_chars,
        )
    return selected


def _build_prompt(
    query: str,
    retrieved_chunks: list[str],
    history: Optional[ConversationHistory] = None,
) -> str:
    """
    Construct the full RAG prompt with context, optional history, and query.

    Args:
        query:            The user's natural-language question.
        retrieved_chunks: Relevant text chunks from the vector store.
        history:          Optional conversation history for multi-turn context.

    Returns:
        Fully formatted prompt string.
    """
    # Enforce context character budget.
    safe_chunks = _truncate_context(retrieved_chunks)

    if not safe_chunks:
        context_block = "[No relevant context was retrieved from the knowledge base.]"
    else:
        sections = [
            f"--- CONTEXT SECTION {i} ---\n{chunk.strip()}"
            for i, chunk in enumerate(safe_chunks, start=1)
        ]
        context_block = "\n\n".join(sections)

    history_block = ""
    if history and len(history) > 0:
        history_block = history.format() + "\n\n"

    prompt = (
        f"CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{'=' * 60}\n"
        f"{context_block}\n"
        f"{'=' * 60}\n\n"
        f"{history_block}"
        f"EMPLOYEE QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    model: genai.GenerativeModel,
    history: Optional[ConversationHistory] = None,
) -> str:
    """
    Format the RAG prompt and call the Gemini API to generate a grounded answer.

    If a ConversationHistory object is provided, the current Q&A pair is
    automatically appended to it after a successful response.

    Args:
        query:            The user's question string.
        retrieved_chunks: Top-k text chunks from the vector store.
        model:            An initialised Gemini GenerativeModel instance.
        history:          Optional ConversationHistory for multi-turn support.

    Returns:
        The model's generated answer as a cleaned string.

    Raises:
        ValueError:   If query is empty.
        RuntimeError: If the Gemini API call fails.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    prompt = _build_prompt(query, retrieved_chunks, history=history)
    logger.debug("RAG prompt:\n%s", prompt)

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("Answer generated (%d chars).", len(answer))

        # Persist this exchange to history for future multi-turn context.
        if history is not None:
            history.add("user", query)
            history.add("assistant", answer)

        return answer

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Gemini API call failed: %s", exc)
        raise RuntimeError(f"Failed to generate answer from Gemini: {exc}") from exc
