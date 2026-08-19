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
    python app.py --debug      # Show retrieved context chunks per query
    python app.py --rebuild    # Force rebuild of FAISS index

Special commands inside the chat loop:
    !help     — Show available commands
    !history  — Print this session's Q&A history
    !stats    — Show vector store statistics
    !save     — Export session Q&A to a timestamped .txt file
    !clear    — Clear conversation history
    exit/quit — Exit the assistant
"""

import argparse
import datetime
import logging
import os
import sys
import time

from dotenv import load_dotenv

from embeddings import chunk_text
from vectorstore import FAISSVectorStore
from rag import initialise_gemini, generate_answer, ConversationHistory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE: str = os.path.join(BASE_DIR, "knowledge.txt")
INDEX_FILE: str = os.path.join(BASE_DIR, "faiss_store", "index.faiss")
CHUNKS_FILE: str = os.path.join(BASE_DIR, "faiss_store", "chunks.json")

TOP_K: int = 3
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour helpers (no external deps)
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"
_DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes if the terminal supports colour."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print(_c("""
╔══════════════════════════════════════════════════════════════╗
║       🤖  ACME CORP — AI FAQ ASSISTANT  🤖                  ║
║    Powered by Google Gemini + FAISS Vector Search           ║
╠══════════════════════════════════════════════════════════════╣
║  Ask about policies, HR, benefits, travel & more.           ║
║  Type  !help  to see all commands.                          ║
╚══════════════════════════════════════════════════════════════╝
""", _CYAN, _BOLD))


def _print_divider() -> None:
    print(_c("\n" + "─" * 65 + "\n", _DIM))


def _print_help() -> None:
    print(_c("""
  Commands
  ──────────────────────────────────────
  !help     Show this help message
  !history  Print session Q&A history
  !stats    Show vector index statistics
  !save     Export session history to file
  !clear    Clear conversation history
  exit/quit Close the assistant
""", _CYAN))


# ---------------------------------------------------------------------------
# Vector store initialisation
# ---------------------------------------------------------------------------

def initialise_vector_store(rebuild: bool = False) -> FAISSVectorStore:
    """
    Load the FAISS index from cache, or build it fresh from knowledge.txt.

    Args:
        rebuild: Force a fresh build even if a cache exists.

    Returns:
        A ready-to-use FAISSVectorStore instance.

    Raises:
        FileNotFoundError: If knowledge.txt is missing.
        ValueError:        If knowledge.txt is empty.
        RuntimeError:      If embedding or index build fails.
    """
    store = FAISSVectorStore()

    if not rebuild and store.load(INDEX_FILE, CHUNKS_FILE):
        print(_c("  FAISS index loaded from cache.\n", _GREEN))
        return store

    if not os.path.isfile(KNOWLEDGE_FILE):
        raise FileNotFoundError(
            f"Knowledge base not found: '{KNOWLEDGE_FILE}'. "
            "Ensure knowledge.txt is in the project root."
        )

    print(_c("  Reading knowledge.txt ...", _YELLOW))
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as fp:
        raw_text = fp.read()

    if not raw_text.strip():
        raise ValueError("knowledge.txt is empty. Add FAQ content and retry.")

    print(_c("  Chunking text ...", _YELLOW))
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(_c(f"  {len(chunks)} chunks created.\n", _GREEN))

    print(_c("  Generating embeddings & building FAISS index ...", _YELLOW))
    print(_c("  (First run: ~30-60 seconds depending on API speed)\n", _DIM))
    store.build_index(chunks)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    store.save(INDEX_FILE, CHUNKS_FILE)
    print(_c("\n  Index saved to disk for future runs.\n", _GREEN))

    return store


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def _save_session(session_log: list[tuple[str, str]]) -> None:
    """
    Export the current session's Q&A pairs to a timestamped .txt file
    in the project directory.

    Args:
        session_log: List of (question, answer) tuples from this session.
    """
    if not session_log:
        print(_c("  Nothing to save — no questions asked yet.\n", _YELLOW))
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(BASE_DIR, f"session_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as fp:
        fp.write("AI FAQ Assistant — Session Export\n")
        fp.write(f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fp.write("=" * 60 + "\n\n")
        for i, (q, a) in enumerate(session_log, 1):
            fp.write(f"Q{i}: {q}\n")
            fp.write(f"A{i}: {a}\n")
            fp.write("-" * 60 + "\n\n")

    print(_c(f"  Session saved to: {os.path.basename(filename)}\n", _GREEN))


def run_chat_loop(
    store: FAISSVectorStore,
    model,
    debug: bool = False,
) -> None:
    """
    Run the interactive CLI question-answering loop with special commands,
    ANSI colours, response timing, and multi-turn conversation history.

    Args:
        store: Initialised FAISSVectorStore.
        model: Initialised Gemini GenerativeModel.
        debug: If True, display retrieved context chunks before each answer.
    """
    history = ConversationHistory(max_turns=5)
    session_log: list[tuple[str, str]] = []  # (question, answer) pairs
    question_count = 0
    blank_streak = 0            # Counts consecutive empty inputs.
    MAX_BLANK_STREAK = 3        # Warn user after this many blanks in a row.

    _print_banner()

    # Print a quick startup summary.
    stats = store.get_stats()
    print(_c(
        f"  Index ready: {stats['total_chunks']} chunks "
        f"| dim={stats['dimension']} | {stats['index_type']}\n",
        _DIM,
    ))

    while True:
        try:
            user_input = input(_c("❓ Question: ", _CYAN, _BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c("\n\n  Goodbye! Have a great day.\n", _GREEN))
            break

        # ---- Special commands -----------------------------------------
        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print(_c("\n  Goodbye! Have a great day.\n", _GREEN))
            break

        if not user_input:
            blank_streak += 1
            if blank_streak >= MAX_BLANK_STREAK:
                print(_c(
                    f"  Tip: type a question, or '!help' for commands, "
                    f"or 'exit' to quit.\n", _YELLOW
                ))
                blank_streak = 0
            else:
                print(_c("  Please enter a question or type !help.\n", _YELLOW))
            continue

        blank_streak = 0  # Reset streak on any real input.

        if user_input.lower() == "!help":
            _print_help()
            continue

        if user_input.lower() == "!clear":
            history.clear()
            session_log.clear()
            question_count = 0
            print(_c("  Conversation history cleared.\n", _GREEN))
            continue

        if user_input.lower() == "!save":
            _save_session(session_log)
            continue

        if user_input.lower() == "!stats":
            stats = store.get_stats()
            labels = {
                "ready":         "Status",
                "total_chunks":  "Indexed Chunks",
                "dimension":     "Embedding Dim",
                "index_type":    "FAISS Index Type",
                "avg_chunk_len": "Avg Chunk Length (chars)",
            }
            print(_c("\n  Vector Store Statistics", _CYAN, _BOLD))
            for key, val in stats.items():
                label = labels.get(key, key)
                display = _c(str(val), _GREEN if val else _RED)
                print(f"    {_c(label, _DIM):<28} {display}")
            print()
            continue

        if user_input.lower() == "!history":
            if not session_log:
                print(_c("  No questions asked yet this session.\n", _YELLOW))
            else:
                print(_c(f"\n  Session History ({len(session_log)} question(s))", _CYAN, _BOLD))
                for i, (q, a) in enumerate(session_log, 1):
                    print(_c(f"\n  [{i}] Q: ", _DIM) + q)
                    preview = a[:200] + ("..." if len(a) > 200 else "")
                    print(_c(f"      A: ", _DIM) + preview)
                print()
            continue

        # Catch unknown ! commands before they reach the search pipeline.
        if user_input.startswith("!"):
            print(_c(
                f"  Unknown command '{user_input}'. Type !help for available commands.\n",
                _YELLOW
            ))
            continue

        # ---- Standard Q&A flow ----------------------------------------
        _print_divider()
        question_count += 1

        try:
            t_start = time.perf_counter()

            # Retrieve relevant chunks with scores.
            results = store.similarity_search_with_scores(user_input, top_k=TOP_K)
            retrieved_chunks = [r.chunk for r in results]

            # Debug view of retrieved context.
            if debug and results:
                print(_c(f"  [DEBUG] {len(results)} context chunk(s) retrieved:\n", _BLUE))
                for i, r in enumerate(results, 1):
                    print(_c(f"  Chunk {i}  (score: {r.score:.4f})", _BLUE))
                    print(_c("  " + "─" * 55, _DIM))
                    preview = r.chunk[:300].replace("\n", "\n  ")
                    print(f"  {preview}")
                    print()

            # Generate grounded answer.
            print(_c("  Answer:\n", _GREEN, _BOLD))
            answer = generate_answer(
                user_input, retrieved_chunks, model, history=history
            )

            elapsed = time.perf_counter() - t_start

            # Print answer.
            print(answer)
            print()
            print(_c(
                f"  [{elapsed:.2f}s | Q#{question_count} | "
                f"history: {len(history)} turn(s) | !save to export]",
                _DIM,
            ))

            # Log to session history.
            session_log.append((user_input, answer))

        except ValueError as exc:
            print(_c(f"  Input error: {exc}", _YELLOW))
        except RuntimeError as exc:
            print(_c(f"  API error: {exc}", _RED))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unexpected error: %s", exc)
            print(_c(f"  Unexpected error: {exc}", _RED))

        _print_divider()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI FAQ Assistant — RAG chatbot powered by Gemini + FAISS."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show retrieved context chunks per query.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of the FAISS index from knowledge.txt.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function: load env, initialise Gemini and FAISS, start chat loop.

    Exit codes:
        0 — Clean exit.
        1 — Configuration or initialisation error.
    """
    args = _get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print(_c("  Debug mode enabled.\n", _BLUE))

    # Load .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(_c(
            "  ERROR: GEMINI_API_KEY not found.\n"
            "  Create a .env file: GEMINI_API_KEY=your_key_here",
            _RED
        ))
        sys.exit(1)

    # Initialise Gemini model
    try:
        print(_c("\n  Initialising Gemini model ...", _YELLOW))
        gemini_model = initialise_gemini(api_key=api_key)
        print(_c("  Gemini ready.\n", _GREEN))
    except (EnvironmentError, RuntimeError) as exc:
        print(_c(f"  Gemini init failed: {exc}", _RED))
        sys.exit(1)

    # Initialise FAISS vector store
    try:
        vector_store = initialise_vector_store(rebuild=args.rebuild)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(_c(f"  Vector store init failed: {exc}", _RED))
        sys.exit(1)

    # Launch chat
    run_chat_loop(vector_store, gemini_model, debug=args.debug)


if __name__ == "__main__":
    main()
