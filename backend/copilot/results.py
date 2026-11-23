"""`make results`: write RESULTS.md and refresh README.md from this run's outputs only.

Inputs (all produced by running the code): data/docs.jsonl, data/chunks.jsonl,
data/copilot.duckdb, data/eval/latest.json (`make eval`), data/eval/tone.json (`make tone`),
data/runlog.json (step timings) and a pytest collection for test counts. Nothing is typed
in by hand: a metric that was not produced is shown as "not measured" in RESULTS.md, and
the matching phrase is removed from the README Highlights.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from copilot.config import REPO_ROOT, get_settings

README = REPO_ROOT / "README.md"
RESULTS = REPO_ROOT / "RESULTS.md"


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0


def _pct(v: float | None) -> str | None:
    return None if v is None else f"{v * 100:.1f}"


def test_counts() -> dict[str, int | None]:
    """Backend tests collected by pytest; frontend unit tests counted by Vitest's files."""
    out: dict[str, int | None] = {"backend": None, "frontend": None}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=REPO_ROOT / "backend",
                       capture_output=True, text=True)  # fmt: skip
    if m := re.search(r"(\d+) tests? collected", r.stdout):
        out["backend"] = int(m.group(1))
    tests = list((REPO_ROOT / "frontend").glob("tests/**/*.test.ts*"))
    if tests:
        out["frontend"] = sum(
            len(re.findall(r"^\s*(?:it|test)\(", p.read_text(encoding="utf-8"), re.M))
            for p in tests
        )
    return out


def collect() -> dict[str, Any]:
    """Gather every measured value."""
    s = get_settings()
    docs = (
        [json.loads(line) for line in s.docs_path.open(encoding="utf-8")]
        if s.docs_path.exists()
        else []
    )
    ev = _json(s.data_dir / "eval" / "latest.json") or {}
    tone = _json(s.data_dir / "eval" / "tone.json") or {}
    runlog = _json(s.data_dir / "runlog.json") or {}
    market = {}
    if s.duckdb_path.exists():
        with duckdb.connect(str(s.duckdb_path), read_only=True) as con:
            for t in ("policy_rate", "corra", "bond_2y", "bond_5y", "bond_10y", "usd_cad"):
                with contextlib.suppress(duckdb.CatalogException):
                    market[t] = con.execute(
                        f"SELECT count(*), min(date), max(date) FROM {t}"
                    ).fetchone()
    m = ev.get("metrics", {})
    g = ev.get("guardrails", {})
    pred = tone.get("prediction", {})
    model = pred.get("logistic_tone_and_decision", {})
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "doc_count": len(docs),
        "docs_by_type": dict(Counter(d["doc_type"] for d in docs)),
        "doc_range": (min(d["date"] for d in docs), max(d["date"] for d in docs)) if docs else None,
        "chunk_count": _count_lines(s.chunks_path),
        "market": market,
        "eval": m,
        "guardrails": g,
        "tone": tone,
        "tone_accuracy": _pct(model.get("accuracy")),
        "tone_f1": model.get("macro_f1"),
        "eval_q_count": m.get("questions"),
        "hit_at_5": _pct(m.get("hit_rate_at_5")),
        "faithfulness": _pct(m.get("faithfulness")),
        "runlog": runlog,
        "tests": test_counts(),
    }


def _v(x: Any, fmt: str = "{}") -> str:
    return "not measured" if x is None else fmt.format(x)


def results_table(d: dict[str, Any]) -> str:
    """The Markdown results table shared by RESULTS.md and the README."""
    m, g, t = d["eval"], d["guardrails"], d["tone"]
    p = t.get("prediction", {})
    rows = [
        ("Documents indexed", _v(d["doc_count"]) + (f" ({', '.join(f'{k} {v}' for k, v in sorted(d['docs_by_type'].items()))})" if d["docs_by_type"] else "")),
        ("Chunks indexed", _v(d["chunk_count"])),
        ("Evaluation questions", _v(m.get("questions"))),
        ("Retrieval hit rate@5", _v(_pct(m.get("hit_rate_at_5")), "{}%")),
        ("MRR@5", _v(m.get("mrr_at_5"))),
        ("Answer faithfulness (LLM judge)", _v(_pct(m.get("faithfulness")), "{}%") + (f" ({m['faithfulness_judged']} answers judged)" if m.get("faithfulness_judged") else "")),
        ("Citation coverage", _v(_pct(m.get("citation_coverage")), "{}%")),
        ("Questions passed", _v(m.get("passed")) + (f" of {m['questions']}" if m.get("questions") else "")),
        ("Latency p50 / p95", f"{_v(m.get('latency_p50_s'), '{} s')} / {_v(m.get('latency_p95_s'), '{} s')}"),
        ("Prompt guardrail checks passed", f"{g['prompt_passed']} of {g['prompt_total']}" if g else "not measured"),
        ("SQL attacks blocked", f"{g['sql_attacks_blocked']} of {g['sql_attacks_total']}" if g else "not measured"),
        ("Announcements in tone index", _v(t.get("announcements"))),
        ("Sentences labelled", _v(t.get("sentences"))),
        ("LLM vs FinBERT agreement (kappa)", f"{_v(_pct(t.get('agreement')), '{}%')} ({_v(t.get('kappa'))})" if t else "not measured"),
        ("Next decision: logistic (tone + decision) accuracy / macro F1", f"{_v(d['tone_accuracy'], '{}%')} / {_v(d['tone_f1'])}"),
        ("Next decision: tone only accuracy / macro F1", f"{_v(_pct(p.get('logistic_tone_only', {}).get('accuracy')), '{}%')} / {_v(p.get('logistic_tone_only', {}).get('macro_f1'))}" if p else "not measured"),
        ("Baseline: same as last decision", f"{_v(_pct(p.get('baseline_persistence', {}).get('accuracy')), '{}%')} / {_v(p.get('baseline_persistence', {}).get('macro_f1'))}" if p else "not measured"),
        ("Baseline: majority class", f"{_v(_pct(p.get('baseline_majority', {}).get('accuracy')), '{}%')} / {_v(p.get('baseline_majority', {}).get('macro_f1'))}" if p else "not measured"),
        ("Backend tests / frontend unit tests", f"{_v(d['tests']['backend'])} / {_v(d['tests']['frontend'])}"),
        ("Model", _v(m.get("model"))),
    ]  # fmt: skip
    return "| Metric | Value |\n| --- | --- |\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


HIGHLIGHTS = [
    ("Built an LLM research copilot (LangGraph, hybrid RAG, reranking) over {doc_count} Bank of Canada "
     "publications, answering with inline citations and safe text-to-SQL over market data.",
     {"doc_count": ("over {doc_count} Bank of Canada publications", "over Bank of Canada publications")}),
    ("Evaluated on a {eval_q_count}-question test set: {hit_at_5}% retrieval hit rate@5 and {faithfulness}% "
     "answer faithfulness, with prompt-injection and SQL guardrails.",
     {"eval_q_count": ("on a {eval_q_count}-question test set", "on a test set"),
      "hit_at_5": ("{hit_at_5}% retrieval hit rate@5 and ", ""),
      "faithfulness": ("{faithfulness}% answer faithfulness, ", "")}),
    ("Created an LLM-based hawkish/dovish tone index of policy announcements, reaching {tone_accuracy}% "
     "accuracy in predicting the next rate decision with logistic regression.",
     {"tone_accuracy": (", reaching {tone_accuracy}% accuracy in predicting the next rate decision with logistic regression",
                        ", and tested whether it predicts the next rate decision with logistic regression")}),
]  # fmt: skip


def highlights(d: dict[str, Any]) -> str:
    """The README Highlights bullets with measured values (phrases dropped when unmeasured)."""
    out = []
    for text, optional in HIGHLIGHTS:
        for key, (phrase, fallback) in optional.items():
            if d.get(key) is None:
                text = text.replace(phrase, fallback)
        out.append("- " + text.format(**{k: d[k] for k in re.findall(r"\{(\w+)\}", text)}))
    return "\n".join(out)


def _replace(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(\n<!-- {name}:end -->)", re.S)
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def write_results() -> str:
    """Write RESULTS.md and update the README sections between the markers."""
    d = collect()
    table = results_table(d)
    steps = d["runlog"]
    timing = (
        "\n".join(f"| `{k}` | {v['seconds']:.0f} s | {v['at']} |" for k, v in steps.items())
        or "| – | – | – |"
    )
    market = (
        "\n".join(f"| {t} | {c:,} | {lo} to {hi} |" for t, (c, lo, hi) in d["market"].items())
        or "| – | – | – |"
    )
    rr = d["doc_range"]
    coverage = f"Documents cover {rr[0]} to {rr[1]}.\n\n" if rr else ""
    RESULTS.write_text(
        f"# Results\n\nGenerated by `make results` on {d['generated']} from this run's outputs. "
        "Nothing here is typed by hand.\n\n"
        f"{table}\n\n"
        f"{coverage}"
        f"## Market data (Bank of Canada Valet API)\n\n| Table | Rows | Dates |\n| --- | ---: | --- |\n{market}\n\n"
        f"## Pipeline timings\n\n| Step | Duration | Finished |\n| --- | ---: | --- |\n{timing}\n\n"
        "## Notes\n\n- The evaluation questions and expected sources were drafted from the documents by an AI "
        "assistant and are marked `reviewed: false` until a person checks them.\n- Faithfulness is judged by "
        "the same local model that answers, which tends to be lenient.\n- The next-decision test uses "
        "expanding-window, time-ordered evaluation; compare it with the baselines above.\n",
        encoding="utf-8",
    )  # fmt: skip
    if README.exists():
        text = README.read_text(encoding="utf-8")
        text = _replace(text, "results", table)
        text = _replace(text, "highlights", highlights(d))
        README.write_text(text, encoding="utf-8")
    return f"wrote {RESULTS.name} and updated {README.name}"


def record_step(step: str, seconds: float) -> None:
    """Append a step timing to data/runlog.json (used by the CLI)."""
    path = get_settings().data_dir / "runlog.json"
    log = _json(path) or {}
    log[step] = {
        "seconds": round(seconds, 1),
        "at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
