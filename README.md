# Sophia ⚡

A fast, open-source AI search and discovery engine built with Python and `uv`.

When you ask a question, Sophia searches the web in real-time, scrapes the actual page content with Trafilatura, picks the most relevant parts with BM25 reranking, and writes a grounded answer with clickable source citations `[1]`, `[2]`.

---

## What it does

1. **Rewrites your question**: Uses conversation history so follow-up questions make sense to a search engine.
2. **Searches DuckDuckGo**: Finds 8–10 real web links without needing a paid search API.
3. **Scrapes web pages**: Uses `trafilatura` to clean articles and remove ads, popups, and navbars.
4. **Picks the best text**: Chunks the content and uses BM25 scoring to find the most relevant sections.
5. **Streams the answer**: Streams the response live using rotating Groq API keys and models with inline citations.
6. **Caches results**: Saves repeated searches in a local SQLite database for 1 hour.

---

## Tech Stack

- **Python 3.13+** with **`uv`** package manager
- **DuckDuckGo (`ddgs`)** for web search
- **Trafilatura & HTTPX** for web scraping
- **Groq & OpenAI SDKs** for fast LLM inference (with automatic key rotation)
- **FastAPI** for Server-Sent Events (SSE) streaming
- **Vanilla HTML/CSS/JS** for a clean, dark-mode web interface

---

## Setup & Installation

### 1. Clone & install dependencies
Make sure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
# Install all dependencies automatically
uv sync
```

### 2. Configure your API keys
Create a `.env` file in the root folder:

```env
# Comma-separated list of Groq API keys (auto-rotated)
GROQ_API_KEYS=your_key_1,your_key_2,your_key_3

# Optional: NVIDIA NIM API key for fallback
NVIDIA_NIM_API_KEY=your_nvidia_key
```

---

## How to Run

### Option 1: Web App (Recommended)
Start the FastAPI server:

```bash
uv run uvicorn sophia.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
👉 **http://127.0.0.1:8000**

- See live progress while it searches and scrapes
- Click any `[1]` citation to highlight the source card
- Click follow-up suggestions to continue the chat

---

### Option 2: Terminal CLI
You can also ask questions directly from the command line:

```bash
# Ask a one-off question
uv run python main.py "Why is DeepSeek R1 significant?"

# Or start an interactive chat session
uv run python main.py
```

---

## Running Tests

```bash
uv run pytest
```

---

## Project Structure

```
sophia/
├── src/sophia/
│   ├── api/          # FastAPI server and streaming endpoints
│   ├── cache/        # SQLite cache for fast repeat queries
│   ├── chunk/        # Text chunking and BM25 reranking
│   ├── engine/       # Core pipeline orchestrator and citation mapper
│   ├── llm/          # Groq key pool and model rotation
│   ├── scrape/       # Parallel web fetcher and Trafilatura extractor
│   ├── search/       # DuckDuckGo search client
│   └── session/      # Multi-turn conversation store
├── static/           # Web UI (HTML, CSS, JS)
├── tests/            # Pytest test suite
├── main.py           # CLI runner
├── pyproject.toml    # Dependencies configuration
└── README.md
```
