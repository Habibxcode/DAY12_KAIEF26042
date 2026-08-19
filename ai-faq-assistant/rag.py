"""
rag.py
------
Retrieval-Augmented Generation (RAG) inference layer.

Responsible for:
  - Initialising the Google Gemini generative model.
  - Constructing a strict, grounded RAG prompt that prevents hallucination.
  - Formatting context from retrieved chunks.
  - Calling the Gemini API and returning a clean text response.
"""

import logging
import os
from typing import Optional

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "gemini-2.5-flash"
FALLBACK_MODEL: str = "gemini-1.5-flash"

# The strict RAG system prompt — instructs the model to stay grounded.
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
"""


# ---------------------------------------------------------------------------
# Gemini Model Initialisation
# ---------------------------------------------------------------------------

def _create_model(model_name: str) -> genai.GenerativeModel:
    """
    Instantiate a Gemini GenerativeModel with the RAG system prompt.

    Args:
        model_name: The Gemini model identifier string.

    Returns:
        A configured GenerativeModel instance.
    """
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=RAG_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.1,      # Low temperature for factual, grounded answers.
            max_output_tokens=800,
        ),
    )


def initialise_gemini(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure the Google Generative AI SDK and return the Gemini model.

    Attempts `DEFAULT_MODEL` first; falls back to `FALLBACK_MODEL` if the
    primary model is unavailable in the current region or subscription tier.

    Args:
        api_key: Optional API key. If None, reads from the environment variable
                 `GEMINI_API_KEY` (which must have been loaded via dotenv).

    Returns:
        An initialised GenerativeModel ready for inference.

    Raises:
        EnvironmentError: If no API key is available.
        RuntimeError:     If both the primary and fallback models fail to load.
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
        f"Failed to initialise both '{DEFAULT_MODEL}' and '{FALLBACK_MODEL}'. "
        "Check your API key and network connection."
    )


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """
    Construct the RAG prompt by injecting retrieved context chunks.

    The prompt separates context sections with clear delimiters to help
    the model distinguish between multiple retrieved passages.

    Args:
        query:            The user's natural-language question.
        retrieved_chunks: List of relevant text chunks from the vector store.

    Returns:
        The fully formatted prompt string.
    """
    if not retrieved_chunks:
        context_block = "[No relevant context was retrieved from the knowledge base.]"
    else:
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_sections.append(
                f"--- CONTEXT SECTION {i} ---\n{chunk.strip()}"
            )
        context_block = "\n\n".join(context_sections)

    prompt = (
        f"CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{'=' * 60}\n"
        f"{context_block}\n"
        f"{'=' * 60}\n\n"
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
) -> str:
    """
    Format the RAG prompt and call the Gemini API to generate an answer.

    The function enforces grounded generation — the model is instructed
    (via system prompt) to answer only from provided context and to
    gracefully refuse if context is insufficient.

    Args:
        query:            The user's question string.
        retrieved_chunks: Top-k text chunks retrieved from the vector store.
        model:            An initialised Gemini GenerativeModel instance.

    Returns:
        The model's generated answer as a cleaned string.

    Raises:
        ValueError:   If `query` is empty.
        RuntimeError: If the API call fails.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    prompt = _build_prompt(query, retrieved_chunks)
    logger.debug("RAG prompt:\n%s", prompt)

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("Answer generated successfully (%d chars).", len(answer))
        return answer
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Gemini API call failed: %s", exc)
        raise RuntimeError(f"Failed to generate answer from Gemini: {exc}") from exc
