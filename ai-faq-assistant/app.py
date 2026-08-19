"""
app.py
------
AI FAQ Assistant — Main entry point.

Orchestrates the full RAG pipeline:
  1. Load environment variables (GEMINI_API_KEY).
  2. Check if a FAISS index cache exists; build it from knowledge.txt if not.
  3. Launch an interactive CLI loop for user Q&A.

Run with:
    python app.py              # Normal mode
    python app.py --debug      # Debug mode (shows retrieved context chunks)
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Local modules
from embeddings import chunk_text
from vectorstore import FAISSVectorStore
from rag import initialise_gemini, generate_answer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE: str = os.path.join(BASE_DIR, "knowledge.txt")
INDEX_FILE: str = os.path.join(BASE_DIR, "faiss_store", "index.faiss")
CHUNKS_FILE: str = os.path.join(BASE_DIR, "faiss_store", "chunks.json")

TOP_K: int = 3          # Number of context chunks to retrieve per query.
CHUNK_SIZE: int = 500   # Characters per chunk.
CHUNK_OVERLAP: int = 50 # Overlap characters between consecutive chunks.

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,   # Suppress verbose INFO in normal mode.
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print a welcoming banner to the terminal."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  ACME CORP — AI FAQ ASSISTANT  🤖               ║
║     Powered by Google Gemini + FAISS Vector Search          ║
╠══════════════════════════════════════════════════════════════╣
║  Ask me anything about company policies, HR, benefits,      ║
║  travel reimbursement, equipment, or code of conduct.       ║
║                                                             ║
║  Commands:  exit | quit  →  close the assistant             ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def _print_divider() -> None:
    print("\n" + "─" * 65 + "\n")


def _get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI FAQ Assistant — RAG chatbot powered by Gemini + FAISS."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: shows retrieved context chunks per query.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the FAISS index even if a cache exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Index initialisation
# ---------------------------------------------------------------------------

def initialise_vector_store(rebuild: bool = False) -> FAISSVectorStore:
    """
    Load the FAISS index from cache or build it from knowledge.txt.

    Args:
        rebuild: If True, skip cache and rebuild the index from scratch.

    Returns:
        A ready-to-use FAISSVectorStore instance.

    Raises:
        FileNotFoundError: If knowledge.txt is missing when rebuilding.
        RuntimeError:      If the index build fails.
    """
    store = FAISSVectorStore()

    # Attempt to load cached index unless rebuild is forced.
    if not rebuild and store.load(INDEX_FILE, CHUNKS_FILE):
        print("✅  FAISS index loaded from cache.\n")
        return store

    # Read the knowledge base.
    if not os.path.isfile(KNOWLEDGE_FILE):
        raise FileNotFoundError(
            f"Knowledge base not found: '{KNOWLEDGE_FILE}'. "
            "Ensure knowledge.txt is in the project root."
        )

    print("📄  Reading knowledge base from knowledge.txt …")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as fp:
        raw_text = fp.read()

    if not raw_text.strip():
        raise ValueError("knowledge.txt is empty. Add FAQ content and retry.")

    # Chunk the text.
    print("✂️   Chunking text …")
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"     → {len(chunks)} chunks created.")

    # Build the FAISS index (this makes embedding API calls).
    print("🔢  Generating embeddings & building FAISS index …")
    print("     (This may take ~30–60 seconds on first run.)\n")
    store.build_index(chunks)

    # Persist to disk.
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    store.save(INDEX_FILE, CHUNKS_FILE)
    print("💾  Index saved to disk for future runs.\n")

    return store


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def run_chat_loop(
    store: FAISSVectorStore,
    model,
    debug: bool = False,
) -> None:
    """
    Run the interactive CLI question-answering loop.

    Args:
        store: Initialised FAISSVectorStore for context retrieval.
        model: Initialised Gemini GenerativeModel for answer generation.
        debug: If True, print the retrieved context chunks before the answer.
    """
    _print_banner()

    while True:
        try:
            user_input = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Goodbye! Have a great day.")
            break

        # Handle exit commands.
        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print("\n👋  Goodbye! Have a great day.")
            break

        # Skip empty input.
        if not user_input:
            print("⚠️  Please enter a question.\n")
            continue

        _print_divider()

        try:
            # Step 1 — Retrieve relevant chunks from the vector store.
            retrieved_chunks = store.similarity_search(user_input, top_k=TOP_K)

            # Optional debug view of retrieved context.
            if debug:
                print(f"🔍  [DEBUG] Retrieved {len(retrieved_chunks)} context chunk(s):\n")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    print(f"  ┌── Chunk {i} ──────────────────────────────────────")
                    print(f"  │  {chunk[:300].replace(chr(10), chr(10) + '  │  ')}")
                    print(f"  └{'─' * 52}\n")

            # Step 2 — Generate a grounded answer via Gemini RAG.
            print("🤖  Answer:\n")
            answer = generate_answer(user_input, retrieved_chunks, model)
            print(answer)

        except ValueError as exc:
            print(f"⚠️  Input error: {exc}")
        except RuntimeError as exc:
            print(f"❌  API error: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unexpected error: %s", exc)
            print(f"❌  An unexpected error occurred: {exc}")

        _print_divider()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main function: loads environment, builds/loads index, starts chat loop.

    Exit codes:
        0  — Clean exit.
        1  — Configuration or initialisation error.
    """
    args = _get_args()

    # Activate debug logging if requested.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🐛  Debug mode enabled.\n")

    # Load .env file.
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "❌  ERROR: GEMINI_API_KEY not found.\n"
            "    Create a .env file in the project root:\n"
            "    GEMINI_API_KEY=your_api_key_here"
        )
        sys.exit(1)

    # Initialise Gemini model.
    try:
        print("🚀  Initialising Gemini model …")
        gemini_model = initialise_gemini(api_key=api_key)
        print("✅  Gemini model ready.\n")
    except (EnvironmentError, RuntimeError) as exc:
        print(f"❌  Gemini initialisation failed: {exc}")
        sys.exit(1)

    # Initialise the vector store (load cache or build fresh).
    try:
        vector_store = initialise_vector_store(rebuild=args.rebuild)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"❌  Vector store initialisation failed: {exc}")
        sys.exit(1)

    # Start the interactive Q&A session.
    run_chat_loop(vector_store, gemini_model, debug=args.debug)


if __name__ == "__main__":
    main()
