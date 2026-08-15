"""FastAPI application for Perplexity Clone with Server-Sent Events (SSE) streaming."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sophia.engine.perplexity import PerplexityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sophia - AI Search Engine API",
    description="Live RAG search engine with Trafilatura scraping, BM25 reranking, and double round-robin LLM key/model rotation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PerplexityEngine()


class QueryRequest(BaseModel):
    query: str = Field(..., description="User search query")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")
    use_cache: bool = Field(default=True, description="Whether to check SQLite cache")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "engine": "running"}


@app.get("/api/models")
async def get_models_status():
    """Return key pool status, rotating models, and live Groq models."""
    pool_status = engine.router.get_pool_status()
    models = engine.router.list_groq_models()
    return {
        "pool": pool_status,
        "active_models_count": len(models),
        "models": models,
        "rotating_models": engine.router.model_pool.models,
    }


@app.get("/api/query/stream")
async def stream_query(
    q: str = Query(..., description="User query text"),
    session_id: str | None = Query(default=None, description="Session ID"),
    use_cache: bool = Query(default=True, description="Use cache"),
    mode: str = Query(default="quick", description="Synthesis mode: 'quick' or 'deep'"),
):
    """Server-Sent Events (SSE) endpoint streaming real-time pipeline events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in engine.run_pipeline(
                user_question=q,
                session_id=session_id,
                use_cache=use_cache,
                mode=mode,
            ):
                payload = {
                    "event_type": event.event_type,
                    "data": (
                        [item.model_dump() if hasattr(item, "model_dump") else item for item in event.data]
                        if isinstance(event.data, list)
                        else (event.data.model_dump() if hasattr(event.data, "model_dump") else event.data)
                    ),
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error("SSE stream error: %s", str(e), exc_info=True)
            err_payload = {"event_type": "error", "data": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/query")
async def json_query(req: QueryRequest):
    """One-shot JSON query endpoint."""
    resp = await engine.ask(user_question=req.query, session_id=req.session_id)
    return resp


@app.get("/api/session/{session_id}")
async def get_session_history(session_id: str):
    """Get multi-turn conversation history for a session."""
    session = engine.session_store.get(session_id)
    if not session:
        return {"session_id": session_id, "turns": []}
    return session


# Mount static files directory for the frontend web application
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Perplexity Clone Running</h1>"
