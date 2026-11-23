"""The SQL tool must only ever run read-only SELECTs over allowlisted tables."""

import duckdb
import pytest

from copilot.agent.sql_tool import UnsafeSQL, run_sql, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE policy_rate",
        "drop table if exists corra",
        "UPDATE policy_rate SET value = 0",
        "DELETE FROM corra",
        "INSERT INTO corra VALUES ('2020-01-01', 1)",
        "CREATE TABLE x AS SELECT 1",
        "ALTER TABLE corra ADD COLUMN y INT",
        "SELECT * FROM policy_rate; DROP TABLE bond_10y",
        "COPY policy_rate TO 'out.csv'",
        "ATTACH 'other.db' AS other",
        "PRAGMA database_list",
        "SELECT * FROM secrets",
        "SELECT * FROM users",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM main.policy_rate",
        "SELECT * FROM read_csv('C:/secrets.csv')",
        "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
        "SELECT getenv('HOME')",
        "SELECT * FROM policy_rate UNION ALL SELECT * FROM duckdb_settings()",
        "WITH x AS (SELECT * FROM secrets) SELECT * FROM x",
    ],
)
def test_unsafe_sql_is_blocked(sql: str) -> None:
    with pytest.raises(UnsafeSQL):
        validate_sql(sql, max_rows=500)


def test_select_gets_a_limit() -> None:
    assert validate_sql("SELECT date, value FROM policy_rate", 500).endswith("LIMIT 500")
    assert validate_sql("SELECT * FROM bond_10y LIMIT 10", 500).endswith("LIMIT 10")
    assert validate_sql("SELECT * FROM bond_10y LIMIT 100000", 500).endswith("LIMIT 500")


def test_ctes_and_joins_over_allowlisted_tables_pass() -> None:
    sql = (
        "WITH m AS (SELECT date_trunc('month', date) AS month, avg(value) AS y FROM bond_10y GROUP BY 1) "
        "SELECT m.month, m.y, p.value FROM m JOIN policy_rate p ON p.date = m.month ORDER BY 1"
    )
    assert "LIMIT" in validate_sql(sql, 500)


def test_run_sql_reads_and_the_connection_is_read_only(data_dir) -> None:
    res = run_sql("SELECT date, value FROM policy_rate ORDER BY date")
    assert res.columns == ["date", "value"] and len(res.rows) == 3
    assert res.sql.endswith("LIMIT 500")
    # Even a statement that slipped past validation could not write: the connection is read-only.
    with (
        duckdb.connect(str(data_dir / "copilot.duckdb"), read_only=True) as con,
        pytest.raises(duckdb.Error),
    ):
        con.execute("DROP TABLE policy_rate")
