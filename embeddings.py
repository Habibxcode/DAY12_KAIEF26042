"""
embeddings.py
-------------
Handles text chunking and embedding generation using Google Gemini's
text-embedding-004 model. Includes sliding-window chunking, batch
embedding with rate-limit handling, and error resilience.
"""

import time
import logging
from typing import Optional

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks using a sliding-window approach.

    The function first attempts to split on paragraph boundaries (double
    newlines) so that semantically coherent blocks are preserved. If a
    paragraph block exceeds `chunk_size` characters, it is further split
    character-by-character with overlap.

    Args:
        text:          Raw input text to be chunked.
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Number of characters to overlap between consecutive
                       chunks to preserve context at boundaries.

    Returns:
        A list of non-empty text chunk strings.

    Raises:
        ValueError: If chunk_size is not positive or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    text = text.strip()
    if not text:
        logger.warning("chunk_text received an empty string; returning [].")
        return []

    # ---- Step 1: split on paragraph boundaries first -----
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph keeps us within chunk_size, append it.
        candidate = (current_chunk + "\n\n" + paragraph).strip()
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Flush current_chunk if non-empty before processing long paragraph.
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # If paragraph itself is larger than chunk_size, split it.
            if len(paragraph) > chunk_size:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end])
                    start += chunk_size - chunk_overlap
            else:
                current_chunk = paragraph

    # Flush any remaining text.
    if current_chunk:
        chunks.append(current_chunk)

    logger.info(
        "chunk_text produced %d chunks from %d characters "
        "(chunk_size=%d, overlap=%d).",
        len(chunks),
        len(text),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------------

def get_embedding(
    text: str,
    model: str = "models/text-embedding-004",
    task_type: str = "RETRIEVAL_DOCUMENT",
    retries: int = 3,
    retry_delay: float = 2.0,
) -> list[float]:
    """
    Generate a float vector embedding for a single input string using the
    Google Gemini embedding API.

    Args:
        text:        The input text to embed.
        model:       The Gemini embedding model identifier.
        task_type:   Embedding task type hint sent to the API.
                     Use "RETRIEVAL_DOCUMENT" for knowledge chunks and
                     "RETRIEVAL_QUERY" for user queries.
        retries:     Number of retry attempts on transient API errors.
        retry_delay: Base delay (seconds) between retries (exponential back-off).

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError:  If the input text is empty.
        RuntimeError: If the embedding API call fails after all retries.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "Embedding attempt %d/%d failed: %s. Retrying in %.1fs…",
                attempt,
                retries,
                exc,
                wait,
            )
            if attempt < retries:
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to generate embedding after {retries} retries. "
        f"Last error: {last_error}"
    )


def get_embeddings_batch(
    texts: list[str],
    model: str = "models/text-embedding-004",
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 5,
    inter_batch_delay: float = 1.0,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts with batching and rate-limit
    handling.

    Texts are embedded in small batches to avoid hitting API rate limits.
    A short delay is inserted between batches.

    Args:
        texts:              List of input strings to embed.
        model:              Gemini embedding model identifier.
        task_type:          Embedding task type hint.
        batch_size:         Number of texts to embed per API call batch.
        inter_batch_delay:  Seconds to wait between batches.

    Returns:
        A list of embedding vectors in the same order as `texts`.

    Raises:
        ValueError:  If `texts` is empty.
        RuntimeError: If any individual embedding call fails.
    """
    if not texts:
        raise ValueError("texts list must not be empty.")

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        logger.info(
            "Embedding batch %d–%d of %d texts…",
            batch_start + 1,
            min(batch_start + batch_size, total),
            total,
        )
        for text in batch:
            embedding = get_embedding(text, model=model, task_type=task_type)
            all_embeddings.append(embedding)

        # Rate-limit pause between batches (not after the last one).
        if batch_start + batch_size < total:
            time.sleep(inter_batch_delay)

    logger.info("Batch embedding complete: %d vectors generated.", len(all_embeddings))
    return all_embeddings
