"""
vectorstore.py
--------------
Encapsulates all FAISS vector store operations: building an index from text
chunks, persisting it to disk, loading a cached index, running similarity
searches with scores, and incrementally adding new chunks.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from embeddings import get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single similarity search result with score metadata."""
    chunk: str
    score: float
    chunk_idx: int

    def __repr__(self) -> str:
        preview = self.chunk[:60].replace("\n", " ")
        return f"SearchResult(score={self.score:.4f}, chunk='{preview}...')"


class FAISSVectorStore:
    """
    Self-contained FAISS vector store using L2-normalised IndexFlatIP
    (cosine similarity) for semantic retrieval.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.chunks: list[str] = []
        self.dim: int = 0

    @property
    def is_ready(self) -> bool:
        """True if the index is built and contains at least one vector."""
        return self.index is not None and self.index.ntotal > 0

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[str]) -> None:
        """
        Generate embeddings for all text chunks and build a FAISS IndexFlatIP.

        Args:
            chunks: Non-empty list of text strings to index.

        Raises:
            ValueError:   If chunks is empty.
            RuntimeError: If embedding API calls fail.
        """
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        logger.info("Building FAISS index from %d chunks...", len(chunks))

        raw_embeddings: list[list[float]] = get_embeddings_batch(
            chunks, task_type="RETRIEVAL_DOCUMENT", show_progress=True
        )
        matrix = np.array(raw_embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self.dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)  # type: ignore[arg-type]
        self.chunks = list(chunks)

        logger.info(
            "FAISS index built: %d vectors, dim=%d.", self.index.ntotal, self.dim
        )

    def add_chunks(self, new_chunks: list[str]) -> None:
        """
        Incrementally embed and add new chunks to an existing index without
        a full rebuild.

        Args:
            new_chunks: New text strings to embed and append.

        Raises:
            RuntimeError: If the index is not initialised.
            ValueError:   If new_chunks is empty.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Cannot add_chunks: build_index() or load() must be called first."
            )
        if not new_chunks:
            raise ValueError("new_chunks must not be empty.")

        logger.info("Adding %d new chunk(s) to existing index...", len(new_chunks))

        raw_embeddings = get_embeddings_batch(
            new_chunks, task_type="RETRIEVAL_DOCUMENT", show_progress=True
        )
        matrix = np.array(raw_embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self.index.add(matrix)  # type: ignore[union-attr]
        self.chunks.extend(new_chunks)

        logger.info("Index updated: %d total vectors.", self.index.ntotal)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, chunks_path: str) -> None:
        """
        Persist the FAISS index and chunk list to disk.

        Raises:
            RuntimeError: If the index is not built.
            OSError:      If writing fails.
        """
        if not self.is_ready:
            raise RuntimeError("Cannot save: build_index() must be called first.")

        for path in (index_path, chunks_path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        faiss.write_index(self.index, index_path)  # type: ignore[arg-type]
        with open(chunks_path, "w", encoding="utf-8") as fp:
            json.dump(self.chunks, fp, ensure_ascii=False, indent=2)

        logger.info("Saved → '%s', '%s'.", index_path, chunks_path)

    def load(self, index_path: str, chunks_path: str) -> bool:
        """
        Load a previously saved FAISS index and chunk list from disk.

        Returns:
            True on success, False if files are missing or corrupt.
        """
        if not os.path.isfile(index_path) or not os.path.isfile(chunks_path):
            logger.info("Cached index not found — will build fresh.")
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as fp:
                self.chunks = json.load(fp)
            self.dim = self.index.d
            logger.info(
                "Loaded from disk: %d vectors, dim=%d.", self.index.ntotal, self.dim
            )
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load vector store: %s.", exc)
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
        score_threshold: float = 0.0,
    ) -> list[str]:
        """
        Return the top-k most relevant chunk strings for a query.

        Args:
            query:           Natural-language question.
            top_k:           Maximum results to return.
            score_threshold: Minimum cosine similarity score (0–1).

        Returns:
            List of matching chunk strings ordered by descending similarity.
        """
        return [r.chunk for r in self.similarity_search_with_scores(
            query, top_k=top_k, score_threshold=score_threshold
        )]

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Like similarity_search but returns SearchResult objects with scores.

        Args:
            query:           Natural-language question.
            top_k:           Maximum results to return.
            score_threshold: Minimum cosine similarity score to include.

        Returns:
            List of SearchResult objects ordered by descending score.

        Raises:
            RuntimeError: If the index is not initialised.
            ValueError:   If query is empty.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Vector store not initialised. Call build_index() or load() first."
            )
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        query_vec = get_embedding(query, task_type="RETRIEVAL_QUERY")
        query_matrix = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_matrix)

        actual_k = min(top_k, self.index.ntotal)  # type: ignore[union-attr]
        scores, indices = self.index.search(query_matrix, actual_k)  # type: ignore[union-attr]

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            score_val = float(score)
            if score_val < score_threshold:
                continue
            logger.debug("Chunk #%d  score=%.4f", idx, score_val)
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=score_val,
                chunk_idx=int(idx),
            ))

        logger.info(
            "similarity_search: %d results for '%s'.", len(results), query[:60]
        )
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a dict of index statistics.

        Returns:
            Keys: ready, total_chunks, dimension, index_type, avg_chunk_len.
        """
        if not self.is_ready:
            return {
                "ready": False, "total_chunks": 0, "dimension": 0,
                "index_type": "N/A", "avg_chunk_len": 0.0,
            }
        avg_len = (
            sum(len(c) for c in self.chunks) / len(self.chunks)
            if self.chunks else 0.0
        )
        return {
            "ready": True,
            "total_chunks": self.index.ntotal,  # type: ignore[union-attr]
            "dimension": self.dim,
            "index_type": type(self.index).__name__,
            "avg_chunk_len": round(avg_len, 1),
        }
