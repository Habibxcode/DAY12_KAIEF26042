"""
embeddings.py
-------------
Handles text chunking and embedding generation using Google Gemini's
text-embedding-004 model. Includes sliding-window chunking, batch
embedding with rate-limit handling, live progress display, and
error resilience.
"""

import re
import sys
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

# Minimum number of characters a chunk must contain to be indexed.
MIN_CHUNK_LENGTH: int = 30


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise raw text before chunking.

    Improvements over raw input:
    - Collapse runs of 3+ blank lines into two (one paragraph break).
    - Strip trailing whitespace from every line.
    - Remove zero-width and non-printable control characters.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string ready for chunking.
    """
    # Remove zero-width spaces and non-printable control chars (keep \n, \t).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]", "", text)
    # Strip trailing whitespace per line.
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines → 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks using a paragraph-aware
    sliding-window approach.

    The function:
    1. Cleans the text (normalises whitespace, removes control chars).
    2. Splits on paragraph boundaries (double newlines) first so
       semantically coherent blocks are preserved.
    3. Falls back to character-level splitting with overlap for paragraphs
       that exceed `chunk_size`.
    4. Filters out chunks shorter than MIN_CHUNK_LENGTH characters.

    Args:
        text:          Raw input text to be chunked.
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Number of characters to overlap between consecutive
                       chunks to preserve context at boundaries.

    Returns:
        A list of non-empty text chunk strings, each at least
        MIN_CHUNK_LENGTH characters long.

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

    text = clean_text(text)
    if not text:
        logger.warning("chunk_text received an empty string; returning [].")
        return []

    # ---- Step 1: split on paragraph boundaries -------------------------
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (current_chunk + "\n\n" + paragraph).strip()
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Flush current buffer before processing the long paragraph.
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            if len(paragraph) > chunk_size:
                # ---- Step 2: character-level split with overlap ---------
                start = 0
                while start < len(paragraph):
                    end = min(start + chunk_size, len(paragraph))
                    chunks.append(paragraph[start:end])
                    start += chunk_size - chunk_overlap
            else:
                current_chunk = paragraph

    # Flush remaining text.
    if current_chunk:
        chunks.append(current_chunk)

    # ---- Step 3: filter very short chunks ------------------------------
    before = len(chunks)
    chunks = [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]
    filtered = before - len(chunks)
    if filtered:
        logger.info("Filtered out %d chunk(s) shorter than %d chars.", filtered, MIN_CHUNK_LENGTH)

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
# Progress Bar (dependency-free)
# ---------------------------------------------------------------------------

def _print_progress(current: int, total: int, width: int = 35) -> None:
    """
    Print an in-place ASCII progress bar to stdout.

    Args:
        current: Number of items completed.
        total:   Total number of items.
        width:   Width of the bar in characters.
    """
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r     [{bar}] {current}/{total} ({pct*100:.0f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


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
        ValueError:   If the input text is empty.
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
            embedding: list[float] = result["embedding"]

            # Sanity-check: embedding must be a non-empty float list.
            if not embedding or not isinstance(embedding[0], float):
                raise ValueError(f"Unexpected embedding format: {type(embedding)}")

            return embedding

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
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts with batching, rate-limit
    handling, and a live progress bar.

    Args:
        texts:              List of input strings to embed.
        model:              Gemini embedding model identifier.
        task_type:          Embedding task type hint.
        batch_size:         Number of texts to embed per batch.
        inter_batch_delay:  Seconds to wait between batches.
        show_progress:      If True, display a live progress bar.

    Returns:
        A list of embedding vectors in the same order as `texts`.

    Raises:
        ValueError:   If `texts` is empty.
        RuntimeError: If any individual embedding call fails.
    """
    if not texts:
        raise ValueError("texts list must not be empty.")

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start: batch_start + batch_size]

        for text in batch:
            embedding = get_embedding(text, model=model, task_type=task_type)
            all_embeddings.append(embedding)

            if show_progress:
                _print_progress(len(all_embeddings), total)

        # Rate-limit pause between batches (skip after the last batch).
        if batch_start + batch_size < total:
            time.sleep(inter_batch_delay)

    logger.info("Batch embedding complete: %d vectors generated.", len(all_embeddings))
    return all_embeddings
