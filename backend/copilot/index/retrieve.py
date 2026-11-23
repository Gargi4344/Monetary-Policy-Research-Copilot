"""Hybrid retrieval with reranking.

Dense (Qdrant) + BM25, merged with reciprocal rank fusion, then a cross-encoder reranker (`BAAI/bge-reranker-base`) picks the final top 5 from the top 20.

Filters: date range (inclusive, ISO dates) and document type, applied to both retrievers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from qdrant_client import models

from copilot.config import get_settings
from copilot.index.store import embed_query, load_bm25, qdrant, tokenize

log = logging.getLogger("retrieve")
RRF_K = 60


@dataclass
class Hit:
    """A retrieved chunk with its scores at each stage."""

    chunk_id: str
    doc_id: str
    title: str
    date: str
    doc_type: str
    url: str
    section: str
    text: str
    rrf: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _passes(
    p: dict[str, Any], date_from: str | None, date_to: str | None, doc_types: list[str] | None
) -> bool:
    if date_from and p["date"] < date_from:
        return False
    if date_to and p["date"] > date_to:
        return False
    return not (doc_types and p["doc_type"] not in doc_types)


def dense_search(
    query: str, n: int, date_from=None, date_to=None, doc_types=None
) -> list[dict[str, Any]]:
    """Top-n chunks by cosine similarity, with payload filters."""
    must: list[models.Condition] = []
    if doc_types:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=doc_types)))
    if date_from or date_to:
        must.append(
            models.FieldCondition(
                key="date", range=models.DatetimeRange(gte=date_from, lte=date_to)
            )
        )
    res = qdrant().query_points(get_settings().collection, query=embed_query(query), limit=n,
                                query_filter=models.Filter(must=must) if must else None, with_payload=True)  # fmt: skip
    return [p.payload for p in res.points]


def bm25_search(
    query: str, n: int, date_from=None, date_to=None, doc_types=None
) -> list[dict[str, Any]]:
    """Top-n chunks by BM25, with the same filters."""
    idx = load_bm25()
    scores = idx["bm25"].get_scores(tokenize(query))
    order = np.argsort(-scores)
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        p = idx["chunks"][i]
        if _passes(p, date_from, date_to, doc_types):
            out.append(p)
            if len(out) >= n:
                break
    return out


@lru_cache
def reranker():
    """The cross-encoder reranker (loaded once)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().rerank_model, device="cpu", max_length=512)


def search(query: str, *, top_k: int | None = None, candidates: int | None = None,
           date_from: str | None = None, date_to: str | None = None,
           doc_types: list[str] | None = None, rerank: bool = True) -> list[Hit]:  # fmt: skip
    """Hybrid search -> RRF -> rerank. Returns `top_k` hits, best first."""
    s = get_settings()
    top_k = top_k or s.retrieve_top_k
    candidates = candidates or s.retrieve_candidates
    t0 = time.perf_counter()
    dense = dense_search(query, candidates, date_from, date_to, doc_types)
    sparse = bm25_search(query, candidates, date_from, date_to, doc_types)
    by_id = {p["chunk_id"]: p for p in [*dense, *sparse]}
    fused = rrf_merge([[p["chunk_id"] for p in dense], [p["chunk_id"] for p in sparse]])
    dense_rank = {p["chunk_id"]: i for i, p in enumerate(dense, 1)}
    bm25_rank = {p["chunk_id"]: i for i, p in enumerate(sparse, 1)}
    pool = sorted(fused, key=lambda c: -fused[c])[:candidates]
    hits = [Hit(**{k: by_id[c][k] for k in ("chunk_id", "doc_id", "title", "date", "doc_type", "url", "section", "text")},
                rrf=round(fused[c], 5), dense_rank=dense_rank.get(c), bm25_rank=bm25_rank.get(c)) for c in pool]  # fmt: skip
    if rerank and hits:
        scores = reranker().predict([(query, h.text) for h in hits], show_progress_bar=False)
        for h, sc in zip(hits, scores, strict=True):
            h.rerank_score = round(float(sc), 4)
        hits.sort(key=lambda h: -(h.rerank_score or 0))
    log.info(
        "search",
        extra={
            "query": query[:200],
            "candidates": len(pool),
            "ms": round((time.perf_counter() - t0) * 1000),
        },
    )
    return hits[:top_k]
