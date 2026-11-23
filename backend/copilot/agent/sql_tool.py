"""query_market_data: text-to-SQL over DuckDB, with layered safety.

1. The SQL is parsed with sqlglot (DuckDB dialect). Exactly one statement, and it must be
   a query (SELECT, optionally with CTEs or UNION). Any INSERT, UPDATE, DELETE, DROP,
   CREATE, ALTER, COPY, ATTACH, PRAGMA, SET or other command anywhere is rejected.
2. Every table must be on the allowlist (CTE names are allowed); table functions that read
   files or URLs (read_csv, read_parquet, glob, httpfs ...) are rejected.
3. A LIMIT is added (or lowered) to SQL_MAX_ROWS.
4. It runs on a read-only DuckDB connection, so even a bypass of 1-3 cannot write.
The final SQL is returned to the user with the rows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from copilot.config import get_settings

log = logging.getLogger("agent.sql")

ALLOWED_TABLES: dict[str, str] = {
    "policy_rate": "date DATE, value DOUBLE -- Bank of Canada target for the overnight rate (%)",
    "corra": "date DATE, value DOUBLE -- Canadian Overnight Repo Rate Average (%)",
    "bond_2y": "date DATE, value DOUBLE -- 2-year Government of Canada benchmark yield (%)",
    "bond_5y": "date DATE, value DOUBLE -- 5-year Government of Canada benchmark yield (%)",
    "bond_10y": "date DATE, value DOUBLE -- 10-year Government of Canada benchmark yield (%)",
    "usd_cad": "date DATE, value DOUBLE -- USD/CAD exchange rate (CAD per USD)",
    "market_daily": "date DATE, policy_rate, corra, bond_2y, bond_5y, bond_10y, usd_cad DOUBLE -- one row per day (NULL when a series has no value that day)",
    "tone_index": "date DATE, title VARCHAR, tone DOUBLE, decision VARCHAR, policy_rate DOUBLE -- one row per rate announcement (decision is hike/hold/cut)",
}  # fmt: skip
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge,
                   exp.Command, exp.Pragma, exp.Set, exp.Copy, exp.Attach, exp.Detach, exp.Use,
                   exp.Transaction, exp.Commit, exp.Rollback, exp.Grant)  # fmt: skip
FORBIDDEN_FUNCTIONS = ("read_", "glob", "scan", "sniff", "query_table", "query", "getenv",
                       "load", "install", "httpfs", "parquet", "csv", "json", "iceberg", "delta")  # fmt: skip


class UnsafeSQL(ValueError):
    """The SQL failed a safety check; the message says which."""


@dataclass
class SqlResult:
    """Rows from a validated query, plus the exact SQL that ran."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    ms: int
    meta: dict[str, Any] = field(default_factory=dict)


def validate_sql(sql: str, max_rows: int | None = None) -> str:
    """Return a safe, LIMIT-ed version of `sql` or raise UnsafeSQL."""
    max_rows = max_rows or get_settings().sql_max_rows
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise UnsafeSQL(f"could not parse the SQL: {str(exc)[:200]}") from exc
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQL(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQL(f"{node.key.upper()} is not allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name:  # a table function such as read_csv(...)
            raise UnsafeSQL("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQL(f"schema-qualified table {table.sql()} is not allowed")
        if name not in ALLOWED_TABLES and name not in ctes:
            raise UnsafeSQL(f"table '{name}' is not on the allowlist")
    for fn in tree.find_all(exp.Func):
        fname = (fn.sql_name() if not isinstance(fn, exp.Anonymous) else fn.name).lower()
        if any(fname.startswith(p) or fname == p for p in FORBIDDEN_FUNCTIONS):
            raise UnsafeSQL(f"function '{fname}' is not allowed")
    limit = tree.args.get("limit")
    current = None
    if limit is not None and isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        current = int(limit.expression.this)
    if current is None or current > max_rows:
        tree = tree.limit(max_rows, copy=True)
    return tree.sql(dialect="duckdb")


def run_sql(sql: str) -> SqlResult:
    """Validate, then execute on a read-only connection."""
    safe = validate_sql(sql)
    t0 = time.perf_counter()
    with duckdb.connect(str(get_settings().duckdb_path), read_only=True,
                        config={"python_enable_replacements": False}) as con:  # fmt: skip
        # No replacement scans: an unknown table name must never resolve to a Python object.
        cur = con.execute(safe)
        columns = [d[0] for d in cur.description]
        rows = [
            [v.isoformat() if hasattr(v, "isoformat") else v for v in r] for r in cur.fetchall()
        ]
    ms = round((time.perf_counter() - t0) * 1000)
    log.info("sql", extra={"sql": safe, "rows": len(rows), "ms": ms})
    return SqlResult(
        sql=safe,
        columns=columns,
        rows=rows,
        truncated=len(rows) >= get_settings().sql_max_rows,
        ms=ms,
    )


def schema_prompt() -> str:
    """Schema description given to the model for text-to-SQL."""
    lines = [f"- {t}({cols})" for t, cols in ALLOWED_TABLES.items()]
    return "DuckDB tables (read-only):\n" + "\n".join(lines)
