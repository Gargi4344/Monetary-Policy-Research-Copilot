"""Run the whole pipeline on committed fixtures (used by CI; no internet for data).

parse saved Bank of Canada pages -> chunk -> embed into embedded Qdrant + BM25 -> hybrid
search; Valet fixture -> DuckDB -> safe SQL. Exits non-zero if any stage misbehaves.
Usage (from backend/):  uv run python ../scripts/pipeline_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
FIXTURES = BACKEND / "tests" / "fixtures"


def main() -> int:
    """Run each stage and print a short report."""
    tmp = Path(tempfile.mkdtemp(prefix="copilot-fixtures-"))
    os.environ["DATA_DIR"] = str(tmp)
    import duckdb

    from copilot.agent.sql_tool import UnsafeSQL, run_sql, validate_sql
    from copilot.index.retrieve import search
    from copilot.index.store import build_indexes
    from copilot.ingest.chunk import chunk_doc
    from copilot.ingest.scrape import parse_page
    from copilot.ingest.valet import SERIES, load_tables

    pages = [
        ("announcement_2024_06_05.html", "https://www.bankofcanada.ca/2024/06/fad-press-release-2024-06-05/", "announcement"),
        ("mpr_2025_04_16.html", "https://www.bankofcanada.ca/publications/mpr/mpr-2025-04-16/", "mpr"),
    ]  # fmt: skip
    docs = [parse_page((FIXTURES / f).read_text(encoding="utf-8"), u, t) for f, u, t in pages]
    assert all(docs), "a fixture page did not parse"
    chunks = [c for d in docs for c in chunk_doc(d)]
    stats = build_indexes(chunks)
    hits = search("Why did the Bank lower its policy rate in June 2024?", top_k=3, rerank=False)
    assert hits and hits[0].doc_type == "announcement", f"unexpected top hit: {hits[:1]}"

    payload = json.loads((FIXTURES / "valet_V39079.json").read_text(encoding="utf-8"))
    empty = {"observations": []}
    with duckdb.connect(str(tmp / "copilot.duckdb")) as con:
        load_tables(con, {t: (payload if s == "V39079" else empty) for t, (s, _) in SERIES.items()})
    res = run_sql("SELECT date, value FROM policy_rate ORDER BY date")
    assert res.rows, "no rows from the SQL tool"
    blocked = 0
    for bad in ("DROP TABLE policy_rate", "SELECT * FROM secrets", "UPDATE policy_rate SET value = 0"):
        try:
            validate_sql(bad)
        except UnsafeSQL:
            blocked += 1
    assert blocked == 3, "an unsafe statement was not blocked"
    print(json.dumps({"documents": len(docs), "chunks": stats["chunks"], "top_hit": hits[0].title,
                      "sql_rows": len(res.rows), "unsafe_blocked": blocked}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
