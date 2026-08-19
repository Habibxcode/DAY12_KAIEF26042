"""
vectorstore.py
--------------
Encapsulates all FAISS vector store operations: building an index from text
chunks, persisting it to disk, loading a cached index, and running
similarity searches. Maintains strict separation from embedding and
generation logic.
"""

import json
import logging
import os
from typing import Optional

import faiss
import numpy as np

from embeddings import get_embedding, get_embeddings_batch

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    A self-contained FAISS vector store that stores text chunks alongside
    their L2-normalised embeddings for cosine-similarity-based retrieval.

    Attributes:
        index:  The live FAISS index (IndexFlatIP after normalisation).
        chunks: List of raw text strings corresponding to index entries.
        dim:    Embedding dimensionality inferred from the first vector.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.chunks: list[str] = []
        self.dim: int = 0

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[str]) -> None:
        """
        Generate embeddings for all text chunks and build a FAISS
        IndexFlatIP index (inner-product on L2-normalised vectors is
        equivalent to cosine similarity).

        Args:
            chunks: A non-empty list of text strings to index.

        Raises:
            ValueError:  If `chunks` is empty.
            RuntimeError: If the embedding API calls fail.
        """
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        logger.info("Building FAISS index from %d chunks…", len(chunks))

        # Generate embeddings for all chunks (with batching + rate-limit handling).
        raw_embeddings: list[list[float]] = get_embeddings_batch(
            chunks, task_type="RETRIEVAL_DOCUMENT"
        )

        # Convert to numpy float32 matrix and L2-normalise for cosine similarity.
        matrix = np.array(raw_embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self.dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # Inner-product index.
        self.index.add(matrix)  # type: ignore[arg-type]
        self.chunks = list(chunks)

        logger.info(
            "FAISS index built: %d vectors, dimensionality=%d.",
            self.index.ntotal,
            self.dim,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, chunks_path: str) -> None:
        """
        Persist the FAISS index and chunk list to disk.

        Args:
            index_path:  File path for the binary FAISS index (e.g. `index.faiss`).
            chunks_path: File path for the JSON chunk list (e.g. `chunks.json`).

        Raises:
            RuntimeError: If the index has not been built yet.
            OSError: If writing to disk fails.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError(
                "Cannot save: build_index() must be called before save()."
            )

        # Ensure parent directories exist.
        for path in (index_path, chunks_path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as fp:
            json.dump(self.chunks, fp, ensure_ascii=False, indent=2)

        logger.info(
            "Vector store saved → index: '%s', chunks: '%s'.",
            index_path,
            chunks_path,
        )

    def load(self, index_path: str, chunks_path: str) -> bool:
        """
        Load a previously saved FAISS index and chunk list from disk.

        Args:
            index_path:  Path to the binary FAISS index file.
            chunks_path: Path to the JSON chunk list file.

        Returns:
            True if both files were found and loaded successfully,
            False otherwise (missing files or load errors).
        """
        if not os.path.isfile(index_path) or not os.path.isfile(chunks_path):
            logger.info(
                "Cached index not found at '%s' / '%s'. Will build fresh.",
                index_path,
                chunks_path,
            )
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as fp:
                self.chunks = json.load(fp)
            self.dim = self.index.d

            logger.info(
                "Vector store loaded from disk: %d vectors, dim=%d.",
                self.index.ntotal,
                self.dim,
            )
            return True

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load vector store: %s. Rebuilding…", exc)
            self.index = None
            self.chunks = []
            return False

    # ------------------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        Embed the query and return the top-k most semantically similar
        text chunks from the index.

        Args:
            query: The user's natural-language question.
            top_k: Number of top results to return.

        Returns:
            A list of up to `top_k` text chunk strings, ordered by
            descending similarity score.

        Raises:
            RuntimeError: If the index has not been built or loaded.
            ValueError:   If `query` is empty.
        """
        if self.index is None:
            raise RuntimeError(
                "Vector store is not initialised. "
                "Call build_index() or load() first."
            )
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        # Embed and normalise the query vector.
        query_vec = get_embedding(query, task_type="RETRIEVAL_QUERY")
        query_matrix = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_matrix)

        # Cap top_k to available index size.
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_matrix, actual_k)  # type: ignore[attr-defined]

        results: list[str] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for no result.
                continue
            logger.debug("  Chunk #%d  score=%.4f", idx, score)
            results.append(self.chunks[idx])

        logger.info(
            "similarity_search returned %d chunks for query: '%s'.",
            len(results),
            query[:60],
        )
        return results
