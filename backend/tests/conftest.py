"""Shared fixtures: an isolated data dir with a small DuckDB, and the fake LLM."""

import os
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app at a temporary data dir with market tables built from fixtures."""
    from copilot.config import get_settings
    from copilot.ingest import valet

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    payloads = {}
    for table, (series, _) in valet.SERIES.items():
        payloads[table] = {
            "observations": [
                {"d": "2024-06-04", series: {"v": "5.0"}},
                {"d": "2024-06-05", series: {"v": "4.75" if table == "policy_rate" else "3.5"}},
                {"d": "2024-07-24", series: {"v": "4.5" if table == "policy_rate" else "3.4"}},
                {"d": "2024-07-25", series: {"v": ""}},
            ]
        }
    with duckdb.connect(str(tmp_path / "copilot.duckdb")) as con:
        valet.load_tables(con, payloads)
    yield tmp_path
    get_settings.cache_clear()
    os.environ.pop("DATA_DIR", None)


@pytest.fixture
def fake_llm():
    """Install a FakeLLM for the test and remove it afterwards."""
    from copilot.llm import FakeLLM, set_llm

    llm = FakeLLM()
    set_llm(llm)
    yield llm
    set_llm(None)
