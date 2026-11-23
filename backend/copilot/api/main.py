"""FastAPI app. OpenAPI docs at /docs.

POST /api/chat streams server-sent events:
  step    - each tool call as it finishes (route, search_documents, query_market_data, make_chart)
  sources - the retrieved passages (numbered as cited)
  sql     - the exact SQL that ran, with columns and rows
  chart   - a chart spec built from the SQL rows
  token   - answer text
  done    - final answer, citations used, found / refused flags, latency, disclaimer
Every question, tool call and latency is logged as one JSON line.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Annotated, Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from copilot.config import get_settings
from copilot.logs import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
log = logging.getLogger("api")

app = FastAPI(title="Monetary Policy Research Copilot API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])  # fmt: skip


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException) -> JSONResponse:
    """Errors as {"error": {"code", "message"}}."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


class ChatIn(BaseModel):
    """A question plus optional retrieval filters."""

    question: str = Field(min_length=3, max_length=1000)
    date_from: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    doc_types: list[str] | None = None


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _hit_json(i: int, h: Any) -> dict[str, Any]:
    return {"n": i, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "date": h.date,
            "doc_type": h.doc_type, "url": h.url, "section": h.section, "text": h.text,
            "rerank_score": h.rerank_score}  # fmt: skip


@app.post("/api/chat")
def chat(body: ChatIn) -> StreamingResponse:
    """Run the agent and stream steps, evidence and the cited answer."""
    from copilot.agent.graph import DISCLAIMER, NOT_FOUND, answer_stream, citations_for, run_agent
    from copilot.llm import get_llm

    def events():
        t0 = time.perf_counter()
        llm = get_llm()
        try:
            state = run_agent(body.question, llm=llm, date_from=body.date_from, date_to=body.date_to,
                              doc_types=body.doc_types)  # fmt: skip
        except Exception as exc:
            log.exception("agent failed")
            yield _sse(
                "error", {"message": f"The model or index is not available ({type(exc).__name__})."}
            )
            return
        for step in state.get("steps", []):
            yield _sse("step", step)
        hits = state.get("hits") or []
        if hits:
            yield _sse("sources", [_hit_json(i, h) for i, h in enumerate(hits, 1)])
        if sql := state.get("sql"):
            yield _sse(
                "sql",
                {
                    "sql": sql.sql,
                    "columns": sql.columns,
                    "rows": sql.rows[:200],
                    "truncated": sql.truncated,
                },
            )
        elif state.get("sql_error"):
            yield _sse("sql", {"error": state["sql_error"]})
        if state.get("chart"):
            yield _sse("chart", state["chart"])
        answer = ""
        try:
            for tok in answer_stream(state, llm):
                answer += tok
                if answer.strip() != NOT_FOUND:
                    yield _sse("token", tok)
        except Exception as exc:
            log.exception("answer failed")
            yield _sse(
                "error", {"message": f"The model failed while answering ({type(exc).__name__})."}
            )
            return
        found = answer.strip() != NOT_FOUND and not state.get("refusal")
        final = (
            answer
            if found or state.get("refusal")
            else "Not found in the sources. Try rephrasing or widening the date range."
        )
        latency = round(time.perf_counter() - t0, 2)
        log.info("chat", extra={"question": body.question[:300], "tools": [s["tool"] for s in state.get("steps", [])],
                                "found": found, "refused": bool(state.get("refusal")), "latency_s": latency,
                                "model": llm.model})  # fmt: skip
        yield _sse("done", {"answer": final, "found": found, "refused": bool(state.get("refusal")),
                            "citations": citations_for(answer, hits) if found else [],
                            "latency_s": latency, "model": llm.model, "disclaimer": DISCLAIMER})  # fmt: skip

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/search")
def search_endpoint(
    q: Annotated[str, Query(min_length=2)],
    date_from: Annotated[str | None, Query(alias="from")] = None,
    date_to: Annotated[str | None, Query(alias="to")] = None,
    doc_type: Annotated[str | None, Query(alias="type")] = None,
    k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[dict[str, Any]]:
    """Ranked chunks with metadata (hybrid search + rerank)."""
    from copilot.index.retrieve import search

    hits = search(
        q, top_k=k, date_from=date_from, date_to=date_to, doc_types=[doc_type] if doc_type else None
    )
    return [
        _hit_json(i, h) | {"dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank, "rrf": h.rrf}
        for i, h in enumerate(hits, 1)
    ]


def _db() -> duckdb.DuckDBPyConnection:
    if not settings.duckdb_path.exists():
        raise HTTPException(503, "No data yet. Run `make data`.")
    # No replacement scans: a missing table must raise, not resolve to a Python object.
    return duckdb.connect(str(settings.duckdb_path), read_only=True,
                          config={"python_enable_replacements": False})  # fmt: skip


@app.get("/api/tone-index")
def tone_index() -> dict[str, Any]:
    """Tone score per announcement with the decision and policy rate, plus the daily policy rate."""
    with _db() as con:
        try:
            rows = con.execute(
                "SELECT date, doc_id, title, url, tone, decision, policy_rate FROM tone_index ORDER BY date"
            ).fetchall()
        except duckdb.CatalogException:
            rows = []
        rate = con.execute("SELECT date_trunc('month', date) AS m, last(value ORDER BY date) FROM policy_rate "
                           "WHERE date >= '2015-01-01' GROUP BY 1 ORDER BY 1").fetchall()  # fmt: skip
    return {"announcements": [dict(zip(("date", "doc_id", "title", "url", "tone", "decision", "policy_rate"), r, strict=True)) for r in rows],
            "policy_rate": [{"date": r[0], "value": r[1]} for r in rate]}  # fmt: skip


@app.get("/api/tone-index/{doc_id}")
def tone_sentences(doc_id: str) -> dict[str, Any]:
    """The announcement's sentences with their labels (most hawkish and dovish first)."""
    with _db() as con:
        try:
            rows = con.execute(
                "SELECT idx, sentence, llm_label, finbert_label FROM tone_sentences WHERE doc_id = ? ORDER BY idx",
                [doc_id],
            ).fetchall()
        except duckdb.CatalogException:
            rows = []
    if not rows:
        raise HTTPException(404, "No tone labels for this announcement.")
    sents = [{"idx": r[0], "sentence": r[1], "llm": r[2], "finbert": r[3]} for r in rows]
    return {"doc_id": doc_id, "hawkish": [s for s in sents if s["llm"] == "hawkish"],
            "dovish": [s for s in sents if s["llm"] == "dovish"], "counts": Counter(s["llm"] for s in sents)}  # fmt: skip


@app.get("/api/eval/latest")
def eval_latest() -> dict[str, Any]:
    """Latest evaluation run (metrics, guardrails, per-question rows)."""
    path = settings.data_dir / "eval" / "latest.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation yet. Run `make eval`.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/sources")
def sources(
    q: str | None = None, doc_type: Annotated[str | None, Query(alias="type")] = None
) -> dict[str, Any]:
    """Indexed documents (filterable), with chunk counts."""
    from copilot.ingest.chunk import load_chunks
    from copilot.ingest.scrape import load_docs

    if not settings.docs_path.exists():
        return {"documents": [], "total": 0, "last_refreshed": None}
    counts = (
        Counter(c.doc_id for c in load_chunks()) if settings.chunks_path.exists() else Counter()
    )
    docs = [
        d
        for d in load_docs()
        if (not doc_type or d.doc_type == doc_type) and (not q or q.lower() in d.title.lower())
    ]
    refreshed = time.strftime("%Y-%m-%d %H:%M", time.localtime(settings.docs_path.stat().st_mtime))
    return {"total": len(docs), "last_refreshed": refreshed, "documents": [
        {"doc_id": d.doc_id, "title": d.title, "date": d.date, "doc_type": d.doc_type, "url": d.url,
         "words": len(d.text.split()), "chunks": counts.get(d.doc_id, 0)} for d in sorted(docs, key=lambda d: d.date, reverse=True)]}  # fmt: skip


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Status, model in use and index size."""
    from copilot.index.store import load_bm25

    try:
        chunks = len(load_bm25()["chunks"])
    except FileNotFoundError:
        chunks = 0
    docs = (
        sum(1 for _ in settings.docs_path.open(encoding="utf-8"))
        if settings.docs_path.exists()
        else 0
    )
    return {"status": "ok", "model": settings.llm_model, "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model, "documents": docs, "chunks": chunks}  # fmt: skip
