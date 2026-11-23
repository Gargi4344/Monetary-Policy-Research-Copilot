"""Optional hand-labelling: measure the LLM tone labels against a person.

`copilot tone-sample` writes 50 random announcement sentences to data/labels/human_tone.csv
with an empty `human_label` column. Fill it with hawkish / dovish / neutral (using the rubric
in copilot/tone/index.py), then `copilot tone-agreement` reports accuracy and Cohen's kappa of
the LLM and FinBERT labels against yours.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

from copilot.config import get_settings
from copilot.tone.index import LABELS, cohen_kappa


def sample_path() -> Path:
    """Where the labelling sheet lives (gitignored data folder)."""
    return get_settings().data_dir / "labels" / "human_tone.csv"


def export_sample(n: int = 50, seed: int = 7) -> Path:
    """Write n random labelled sentences for a person to label blind."""
    with duckdb.connect(str(get_settings().duckdb_path), read_only=True) as con:
        con.execute(f"SELECT setseed({seed / 100})")
        rows = con.execute(
            "SELECT doc_id, idx, sentence FROM tone_sentences ORDER BY random() LIMIT ?", [n]
        ).fetchall()
    path = sample_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "idx", "sentence", "human_label"])
        w.writerows([[d, i, s, ""] for d, i, s in rows])
    return path


def agreement() -> dict[str, Any]:
    """Accuracy and kappa of the LLM and FinBERT labels against the filled-in sheet."""
    with sample_path().open(encoding="utf-8") as f:
        human = {
            (r["doc_id"], int(r["idx"])): r["human_label"].strip().lower()
            for r in csv.DictReader(f)
        }
    human = {k: v for k, v in human.items() if v in LABELS}
    if not human:
        return {"labelled": 0}
    with duckdb.connect(str(get_settings().duckdb_path), read_only=True) as con:
        rows = con.execute(
            "SELECT doc_id, idx, llm_label, finbert_label FROM tone_sentences"
        ).fetchall()
    pairs = [(human[(d, i)], llm, fb) for d, i, llm, fb in rows if (d, i) in human]
    h = [p[0] for p in pairs]
    out: dict[str, Any] = {"labelled": len(pairs)}
    for name, col in (("llm", 1), ("finbert", 2)):
        pred = [p[col] or "neutral" for p in pairs]
        out[name] = {"accuracy": round(sum(a == b for a, b in zip(h, pred, strict=True)) / len(h), 3),
                     "kappa": round(cohen_kappa(h, pred), 3)}  # fmt: skip
    return out
