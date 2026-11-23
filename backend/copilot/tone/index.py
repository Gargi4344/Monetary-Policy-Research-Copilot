"""Hawkish/dovish tone index of rate announcements, and whether tone predicts the next decision.

Pipeline
1. Split each interest rate announcement into sentences (boilerplate removed).
2. Label each sentence with the LLM using a fixed rubric (batches of up to 20, JSON output).
3. Second baseline: FinBERT (ProsusAI/finbert) sentiment, mapped positive -> hawkish,
   negative -> dovish, neutral -> neutral. FinBERT measures sentiment, not policy stance, so
   this is a rough proxy; agreement is reported as raw agreement and Cohen's kappa.
4. Tone score per announcement = (hawkish - dovish) / sentences, in [-1, 1].
5. The decision at each announcement (hike / hold / cut) comes from the Valet policy rate
   series, not from the text. A logistic regression predicts the NEXT decision from this
   announcement's tone (and the current decision), evaluated with expanding-window,
   time-ordered cross-validation, against majority-class and "same as last time" baselines.
Results are cached in DuckDB (`tone_sentences`, `tone_index`) so re-runs only label new text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np

from copilot.config import get_settings
from copilot.ingest.scrape import Doc
from copilot.llm import LLM, get_llm, parse_json

log = logging.getLogger("tone")
LABELS = ("hawkish", "dovish", "neutral")
BATCH = 20
BOILERPLATE = re.compile(
    r"(information note|next scheduled date|for more information|media relations|"
    r"monetary policy report \(mpr\) will be published|full update of the outlook|^ottawa)",
    re.I,
)
RUBRIC = (
    "You classify sentences from Bank of Canada interest rate announcements by monetary policy "
    "stance. Labels:\n"
    "- hawkish: points toward higher rates or tighter policy: inflation too high or persistent, "
    "strong demand, overheating, upside inflation risks, need to raise rates or keep them high, "
    "quantitative tightening.\n"
    "- dovish: points toward lower rates or easier policy: weak growth, slack, rising "
    "unemployment, inflation falling or below target, downside risks, rate cuts, support for the "
    "economy, quantitative easing.\n"
    "- neutral: factual description without a policy lean, procedural statements, or balanced "
    "sentences.\n"
    'Reply with JSON only: {"labels": [one label per sentence, in order]}.'
)


def split_sentences(text: str) -> list[str]:
    """Sentences of at least 6 words, without announcement boilerplate."""
    flat = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", flat)
    return [p.strip() for p in parts if len(p.split()) >= 6 and not BOILERPLATE.search(p)]


def label_batch(llm: LLM, sentences: list[str]) -> list[str]:
    """LLM labels for one batch; anything unparseable becomes neutral (counted in logs)."""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Classify these {len(sentences)} sentences:\n{numbered}"}]  # fmt: skip
    try:
        labels = parse_json(
            llm.complete(msgs, json_mode=True, max_tokens=60 + 8 * len(sentences))
        ).get("labels", [])
    except Exception as exc:
        log.warning("label batch failed", extra={"error": str(exc)[:200]})
        labels = []
    labels = [str(x).lower().strip() for x in labels]
    if len(labels) != len(sentences):
        log.warning("label count mismatch", extra={"expected": len(sentences), "got": len(labels)})
        labels = (labels + ["neutral"] * len(sentences))[: len(sentences)]
    return [x if x in LABELS else "neutral" for x in labels]


FINBERT_MAP = {"positive": "hawkish", "negative": "dovish", "neutral": "neutral"}


def finbert_labels(sentences: list[str]) -> list[str]:
    """FinBERT sentiment mapped to stance (a proxy baseline)."""
    from transformers import pipeline

    clf = pipeline(
        "text-classification", model=get_settings().finbert_model, device=-1, truncation=True
    )
    return [FINBERT_MAP[r["label"].lower()] for r in clf(sentences, batch_size=32)]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two label lists."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b))


def tone_score(labels: list[str]) -> float:
    """(hawkish - dovish) / n, in [-1, 1]."""
    if not labels:
        return 0.0
    return (labels.count("hawkish") - labels.count("dovish")) / len(labels)


def decision_at(con: duckdb.DuckDBPyConnection, day: str) -> tuple[str, float | None]:
    """Hike / hold / cut from the policy rate just before vs just after the announcement day."""
    before = con.execute(
        "SELECT value FROM policy_rate WHERE date < ?::DATE ORDER BY date DESC LIMIT 1", [day]
    ).fetchone()
    after = con.execute(
        "SELECT value FROM policy_rate WHERE date > ?::DATE ORDER BY date LIMIT 1", [day]
    ).fetchone()
    if not before or not after:
        return "unknown", after[0] if after else None
    diff = round(after[0] - before[0], 4)
    return ("hike" if diff > 0 else "cut" if diff < 0 else "hold"), after[0]


@dataclass
class ToneRun:
    """Everything `build_tone_index` produced (written to DuckDB and RESULTS.md)."""

    announcements: int
    sentences: int
    llm_distribution: dict[str, int]
    agreement: float
    kappa: float
    prediction: dict[str, Any]


def predict_next_decision(rows: list[dict[str, Any]], min_train: int = 24) -> dict[str, Any]:
    """Expanding-window evaluation: train on announcements before t, predict decision t+1.

    Features: this announcement's tone and its own decision (one-hot). Baselines: majority
    class of the training window, and persistence (next decision = this decision).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    order = sorted((r for r in rows if r["decision"] != "unknown"), key=lambda r: r["date"])
    cats = ["cut", "hold", "hike"]

    def feats(r: dict[str, Any]) -> list[float]:
        return [r["tone"]] + [1.0 if r["decision"] == c else 0.0 for c in cats]

    X = np.array([feats(r) for r in order[:-1]])
    y = np.array([order[i + 1]["decision"] for i in range(len(order) - 1)])
    preds, majority, persistence, truth = [], [], [], []
    for t in range(min_train, len(y)):
        train_y = y[:t]
        if len(set(train_y)) < 2:
            pred = train_y[0]
        else:
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(X[:t], train_y)
            pred = model.predict(X[t : t + 1])[0]
        preds.append(pred)
        majority.append(max(set(train_y), key=list(train_y).count))
        persistence.append(order[t]["decision"])
        truth.append(y[t])

    def score(p: list[str]) -> dict[str, float]:
        return {"accuracy": round(accuracy_score(truth, p), 3),
                "macro_f1": round(f1_score(truth, p, average="macro", zero_division=0), 3)}  # fmt: skip

    tone_only = []
    for t in range(min_train, len(y)):
        train_y = y[:t]
        if len(set(train_y)) < 2:
            tone_only.append(train_y[0])
            continue
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[:t, :1], train_y)
        tone_only.append(m.predict(X[t : t + 1, :1])[0])
    return {
        "test_announcements": len(truth),
        "train_minimum": min_train,
        "class_counts": {c: int((np.array(truth) == c).sum()) for c in cats},
        "logistic_tone_and_decision": score(preds),
        "logistic_tone_only": score(tone_only),
        "baseline_majority": score(majority),
        "baseline_persistence": score(persistence),
    }


def build_tone_index(docs: list[Doc], llm: LLM | None = None, use_finbert: bool = True) -> ToneRun:
    """Label sentences (cached), score announcements, write `tone_index`, test prediction."""
    llm = llm or get_llm()
    s = get_settings()
    anns = sorted((d for d in docs if d.doc_type == "announcement"), key=lambda d: d.date)
    with duckdb.connect(str(s.duckdb_path)) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS tone_sentences (
            doc_id VARCHAR, idx INTEGER, sentence VARCHAR, llm_label VARCHAR, finbert_label VARCHAR,
            model VARCHAR, PRIMARY KEY (doc_id, idx))""")
        rows = []
        for n, d in enumerate(anns, 1):
            sents = split_sentences(d.text)
            cached = con.execute("SELECT idx, llm_label, finbert_label FROM tone_sentences WHERE doc_id = ? AND model = ?",
                                 [d.doc_id, llm.model]).fetchall()  # fmt: skip
            if len(cached) != len(sents):
                labels = [
                    lab
                    for i in range(0, len(sents), BATCH)
                    for lab in label_batch(llm, sents[i : i + BATCH])
                ]
                fb = finbert_labels(sents) if use_finbert else [None] * len(sents)
                con.execute("DELETE FROM tone_sentences WHERE doc_id = ?", [d.doc_id])
                con.executemany("INSERT INTO tone_sentences VALUES (?, ?, ?, ?, ?, ?)",
                                [(d.doc_id, i, st, lab, f, llm.model) for i, (st, lab, f) in enumerate(zip(sents, labels, fb, strict=True))])  # fmt: skip
                log.info(
                    "labelled",
                    extra={"n": n, "of": len(anns), "date": d.date, "sentences": len(sents)},
                )
            labs = [
                r[0]
                for r in con.execute(
                    "SELECT llm_label FROM tone_sentences WHERE doc_id = ? ORDER BY idx", [d.doc_id]
                ).fetchall()
            ]
            decision, rate = decision_at(con, d.date)
            rows.append({"date": d.date, "doc_id": d.doc_id, "title": d.title, "url": d.url,
                         "tone": round(tone_score(labs), 4), "decision": decision, "policy_rate": rate})  # fmt: skip
        con.execute(
            "CREATE OR REPLACE TABLE tone_index (date DATE, doc_id VARCHAR, title VARCHAR, url VARCHAR, tone DOUBLE, decision VARCHAR, policy_rate DOUBLE)"
        )
        con.executemany("INSERT INTO tone_index VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(r["date"], r["doc_id"], r["title"], r["url"], r["tone"], r["decision"], r["policy_rate"]) for r in rows])  # fmt: skip
        pairs = con.execute(
            "SELECT llm_label, finbert_label FROM tone_sentences WHERE finbert_label IS NOT NULL AND model = ?",
            [llm.model],
        ).fetchall()
        dist = dict(
            con.execute(
                "SELECT llm_label, count(*) FROM tone_sentences WHERE model = ? GROUP BY 1",
                [llm.model],
            ).fetchall()
        )
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    return ToneRun(announcements=len(rows), sentences=sum(dist.values()), llm_distribution=dist,
                   agreement=round(float(np.mean([x == y for x, y in pairs])), 3) if pairs else 0.0,
                   kappa=round(cohen_kappa(a, b), 3) if pairs else 0.0,
                   prediction=predict_next_decision(rows))  # fmt: skip
