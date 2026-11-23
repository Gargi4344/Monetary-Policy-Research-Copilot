"""Bank of Canada Valet API -> DuckDB market tables.

Series (all public, no key): policy rate target, CORRA, Government of Canada benchmark
bond yields (2, 5, 10 years) and the USD/CAD exchange rate. Each becomes a tidy table
(`date`, `value`) plus a combined `market_daily` view for the SQL tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import httpx

from copilot.config import get_settings

log = logging.getLogger("ingest.valet")
VALET = "https://www.bankofcanada.ca/valet/observations/{series}/json"

# table name -> (Valet series id, description)
SERIES: dict[str, tuple[str, str]] = {
    "policy_rate": ("V39079", "Target for the overnight rate (policy rate), percent"),
    "corra": ("AVG.INTWO", "Canadian Overnight Repo Rate Average (CORRA), percent"),
    "bond_2y": ("BD.CDN.2YR.DQ.YLD", "Government of Canada 2-year benchmark bond yield, percent"),
    "bond_5y": ("BD.CDN.5YR.DQ.YLD", "Government of Canada 5-year benchmark bond yield, percent"),
    "bond_10y": (
        "BD.CDN.10YR.DQ.YLD",
        "Government of Canada 10-year benchmark bond yield, percent",
    ),
    "usd_cad": ("FXUSDCAD", "US dollar in Canadian dollars, daily exchange rate"),
}
START = "2010-01-01"


def parse_observations(payload: dict, series: str) -> list[tuple[str, float]]:
    """(date, value) rows from a Valet response; blanks are skipped."""
    rows = []
    for obs in payload.get("observations", []):
        v = obs.get(series, {}).get("v")
        if v not in (None, ""):
            rows.append((obs["d"], float(v)))
    return rows


def fetch(series: str, cache_dir: Path) -> dict:
    """Fetch one series from Valet (cached on disk for the day)."""
    cache = cache_dir / f"valet-{series}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    s = get_settings()
    r = httpx.get(VALET.format(series=series), params={"start_date": START},
                  headers={"User-Agent": s.user_agent}, timeout=60)  # fmt: skip
    r.raise_for_status()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text, encoding="utf-8")
    return r.json()


def load_tables(con: duckdb.DuckDBPyConnection, payloads: dict[str, dict]) -> dict[str, int]:
    """Create the market tables from Valet payloads keyed by table name."""
    counts = {}
    for table, (series, desc) in SERIES.items():
        rows = parse_observations(payloads[table], series)
        con.execute(f"CREATE OR REPLACE TABLE {table} (date DATE PRIMARY KEY, value DOUBLE)")
        if rows:  # a series with no observations still gets an (empty) table
            con.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
        con.execute(f"COMMENT ON TABLE {table} IS '{desc.replace(chr(39), '')}'")
        counts[table] = len(rows)
    con.execute("""
        CREATE OR REPLACE VIEW market_daily AS
        SELECT d.date, p.value AS policy_rate, c.value AS corra, b2.value AS bond_2y,
               b5.value AS bond_5y, b10.value AS bond_10y, fx.value AS usd_cad
        FROM (SELECT date FROM bond_10y UNION SELECT date FROM policy_rate) d
        LEFT JOIN policy_rate p USING (date) LEFT JOIN corra c USING (date)
        LEFT JOIN bond_2y b2 USING (date) LEFT JOIN bond_5y b5 USING (date)
        LEFT JOIN bond_10y b10 USING (date) LEFT JOIN usd_cad fx USING (date)
    """)
    return counts


def ingest_market_data() -> dict[str, int]:
    """Download all series and (re)build the DuckDB market tables."""
    s = get_settings()
    payloads = {table: fetch(series, s.cache_dir) for table, (series, _) in SERIES.items()}
    s.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(s.duckdb_path)) as con:
        counts = load_tables(con, payloads)
    log.info("market data loaded", extra=counts)
    return counts
