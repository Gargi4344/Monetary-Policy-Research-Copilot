"""Build and load the search indexes.

* Dense: `BAAI/bge-small-en-v1.5` embeddings (384 dims, normalised) in Qdrant. Qdrant runs
  in embedded mode (files under data/qdrant/), so no server or container is needed; the
  same code works against a Qdrant server by passing a URL.
* Sparse: BM25 (rank_bm25) over lower-cased word tokens, pickled to data/bm25.pkl.
"""

from __future__ import annotations

import atexit
import logging
import pickle
import re
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from copilot.config import get_settings
from copilot.ingest.chunk import Chunk

log = logging.getLogger("index")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "how",
        "did",
        "does",
        "do",
        "which",
        "who",
        "when",
        "why",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25 (numbers such as 2.75 stay whole)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


@lru_cache
def embedder():
    """The sentence-transformers embedding model (loaded once)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embeddings for passages (normalised, so dot product = cosine)."""
    return (
        embedder()
        .encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        .tolist()
    )


def embed_query(text: str) -> list[float]:
    """Embedding for a query, with the instruction prefix bge models expect."""
    return embedder().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()


_client: QdrantClient | None = None


def qdrant() -> QdrantClient:
    """Embedded Qdrant client (one per process: the local store takes a file lock)."""
    global _client
    if _client is None:
        path = get_settings().qdrant_path
        path.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(path))
        atexit.register(_client.close)  # close before interpreter shutdown (avoids a noisy warning)
    return _client


def payload(c: Chunk) -> dict[str, Any]:
    """Metadata stored with each vector (and returned with search hits)."""
    return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "url": c.url, "doc_type": c.doc_type,
            "title": c.title, "date": c.date, "section": c.section, "text": c.text}  # fmt: skip


def build_indexes(chunks: list[Chunk]) -> dict[str, int]:
    """(Re)create the Qdrant collection and the BM25 index from chunks."""
    s = get_settings()
    client = qdrant()
    vectors = embed_passages([c.text for c in chunks])
    dim = len(vectors[0])
    if client.collection_exists(s.collection):
        client.delete_collection(s.collection)
    client.create_collection(
        s.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    client.upsert(s.collection, points=[
        models.PointStruct(id=i, vector=v, payload=payload(c)) for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ])  # fmt: skip
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    with s.bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": [payload(c) for c in chunks]}, f)
    load_bm25.cache_clear()
    log.info("indexed", extra={"chunks": len(chunks), "dims": dim})
    return {"chunks": len(chunks), "dims": dim}


@lru_cache
def load_bm25() -> dict[str, Any]:
    """The pickled BM25 index and the chunk payloads in index order."""
    with get_settings().bm25_path.open("rb") as f:
        return pickle.load(f)
