"""Chunking, fusion, tone scoring, prediction, JSON parsing and results formatting."""

import duckdb

from copilot.index.retrieve import rrf_merge
from copilot.index.store import tokenize
from copilot.ingest.chunk import chunk_doc, is_heading, sections
from copilot.ingest.scrape import Doc, classify_url, date_from_url
from copilot.llm import FakeLLM, parse_json
from copilot.tone.index import (
    decision_at,
    label_batch,
    predict_next_decision,
    split_sentences,
    tone_score,
)


def make_doc(text: str) -> Doc:
    return Doc(doc_id="abc", url="https://x/2024/06/fad-press-release-2024-06-05/", doc_type="announcement",
               title="Bank of Canada reduces policy rate", date="2024-06-05", text=text)  # fmt: skip


# --- scraping helpers -------------------------------------------------------------------


def test_classify_urls_by_type_and_year() -> None:
    b = "https://www.bankofcanada.ca"
    assert classify_url(f"{b}/2024/06/fad-press-release-2024-06-05/", 2015) == "announcement"
    assert classify_url(f"{b}/2010/06/fad-press-release-2010-06-01/", 2015) is None
    assert classify_url(f"{b}/2025/10/opening-statement-2025-10-29/", 2015) == "opening_statement"
    assert (
        classify_url(
            f"{b}/2024/02/summary-governing-council-deliberations-fixed-announcement-date-january-24-2024/",
            2015,
        )
        == "deliberations"
    )
    assert classify_url(f"{b}/publications/mpr/mpr-2025-04-16/", 2015) == "mpr"
    assert classify_url(f"{b}/publications/mpr/mpr-2025-04-16/projections/", 2015) is None
    assert classify_url(f"{b}/2024/06/some-speech/", 2015) is None


def test_date_from_url() -> None:
    assert date_from_url("https://x/2024/12/fad-press-release-2024-12-11/") == "2024-12-11"
    assert date_from_url("https://x/2015/04/opening-statement-150415/") == "2015-04-01"


# --- chunking ---------------------------------------------------------------------------


def test_headings_and_sections() -> None:
    assert is_heading("Global economy") and not is_heading("Inflation eased to 2.4% in June.")
    secs = sections(
        "Global economy\nGrowth slowed.\nMore text.\nCanadian outlook\nInflation eased."
    )
    assert [h for h, _ in secs] == ["Global economy", "Canadian outlook"]


def test_chunks_respect_size_overlap_and_keep_metadata() -> None:
    para = " ".join(f"w{i}" for i in range(300))
    doc = make_doc(f"Overview\n{para}\n{para}\n{para}")
    chunks = chunk_doc(doc, tokens=500, overlap=0.15)
    assert len(chunks) >= 2
    size = int(500 * 0.75)
    body = [c.text.split("\n")[-1].split() for c in chunks]  # header and section lines removed
    assert all(len(b) <= size for b in body)
    first, second = body[0], body[1]
    assert first[-int(size * 0.15) :] == second[: int(size * 0.15)]
    assert {c.section for c in chunks} == {"Overview"} and chunks[0].date == "2024-06-05"
    assert chunks[0].title == doc.title and chunks[0].url == doc.url
    assert chunks[0].text.startswith(
        "Bank of Canada reduces policy rate — 5 June 2024 (interest rate announcement)\nOverview\n"
    )


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_doc(make_doc("The Bank held its policy rate at 5%. Inflation is easing."))
    assert len(chunks) == 1


# --- retrieval --------------------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers() -> None:
    scores = rrf_merge([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] == scores["b"] > scores["c"] > 0 and scores["c"] == scores["d"]


def test_tokenize_keeps_rates_and_drops_stopwords() -> None:
    assert tokenize("The rate is 2.75% in 2025") == ["rate", "2.75", "2025"]


# --- tone ------------------------------------------------------------------------------


def test_split_sentences_drops_short_and_boilerplate() -> None:
    text = ("The Bank of Canada today held its target for the overnight rate at 5%. Inflation remains too high. "
            "Information note: The next scheduled date for announcing the overnight rate target is July 24, 2024.")  # fmt: skip
    assert split_sentences(text) == [
        "The Bank of Canada today held its target for the overnight rate at 5%."
    ]


def test_tone_score() -> None:
    assert tone_score(["hawkish", "hawkish", "dovish", "neutral"]) == 0.25
    assert tone_score([]) == 0.0


def test_label_batch_validates_model_output() -> None:
    llm = FakeLLM(rules={"Classify": '{"labels": ["hawkish", "DOVISH", "banana"]}'})
    assert label_batch(llm, ["a b c d e f", "a b c d e f", "a b c d e f"]) == [
        "hawkish",
        "dovish",
        "neutral",
    ]
    short = FakeLLM(rules={"Classify": '{"labels": ["hawkish"]}'})
    assert label_batch(short, ["x", "y"]) == ["hawkish", "neutral"]  # count mismatch is padded


def test_decision_from_policy_rate(data_dir) -> None:
    with duckdb.connect(str(data_dir / "copilot.duckdb")) as con:
        assert decision_at(con, "2024-06-05")[0] == "cut"  # 5.0 before, 4.5 after
        assert decision_at(con, "2030-01-01")[0] == "unknown"


def test_prediction_reports_model_and_baselines() -> None:
    import random

    random.seed(1)
    rows, decision = [], "hold"
    for i in range(60):
        tone = random.uniform(-1, 1)
        rows.append({"date": f"2020-{i:03d}", "tone": tone, "decision": decision})
        decision = (
            "hike" if tone > 0.4 else "cut" if tone < -0.4 else "hold"
        )  # next decision follows tone
    p = predict_next_decision(rows, min_train=20)
    assert p["test_announcements"] == 39
    assert set(p) >= {
        "logistic_tone_and_decision",
        "logistic_tone_only",
        "baseline_majority",
        "baseline_persistence",
    }
    assert p["logistic_tone_only"]["accuracy"] > p["baseline_majority"]["accuracy"]


# --- misc -------------------------------------------------------------------------------


def test_parse_json_tolerates_fences_and_text() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"sql": "SELECT 1"} done') == {"sql": "SELECT 1"}


def test_highlights_drop_unmeasured_phrases() -> None:
    from copilot.results import highlights

    d = {
        "doc_count": 300,
        "eval_q_count": 30,
        "hit_at_5": "90.0",
        "faithfulness": None,
        "tone_accuracy": None,
    }
    text = highlights(d)
    assert "over 300 Bank of Canada publications" in text and "90.0% retrieval hit rate@5" in text
    assert (
        "{" not in text
        and "faithfulness" not in text.split("\n")[1].split("rate@5")[1].split(",")[0]
    )
    assert "reaching" not in text


# --- fixtures from the real site (no network in CI) -------------------------------------


def test_parse_real_announcement_page() -> None:
    from copilot.ingest.scrape import parse_page
    from tests.conftest import FIXTURES

    url = "https://www.bankofcanada.ca/2024/06/fad-press-release-2024-06-05/"
    doc = parse_page(
        (FIXTURES / "announcement_2024_06_05.html").read_text(encoding="utf-8"), url, "announcement"
    )
    assert doc is not None and doc.date == "2024-06-05" and doc.doc_type == "announcement"
    assert "reduces policy rate" in doc.title and "overnight rate" in doc.text
    mpr = parse_page((FIXTURES / "mpr_2025_04_16.html").read_text(encoding="utf-8"),
                     "https://www.bankofcanada.ca/publications/mpr/mpr-2025-04-16/", "mpr")  # fmt: skip
    assert mpr is not None and mpr.date == "2025-04-16" and "Overview" in mpr.text


def test_parse_valet_fixture() -> None:
    import json

    from copilot.ingest.valet import parse_observations
    from tests.conftest import FIXTURES

    rows = parse_observations(
        json.loads((FIXTURES / "valet_V39079.json").read_text(encoding="utf-8")), "V39079"
    )
    assert rows and all(isinstance(v, float) for _, v in rows)
    assert dict(rows)["2024-06-04"] == 5.0 and dict(rows)["2024-06-06"] == 4.75


def test_schedule_notices_and_navigation_pages_are_skipped() -> None:
    from copilot.ingest.scrape import parse_page

    b = "https://www.bankofcanada.ca"
    assert classify_url(f"{b}/2026/02/publication-summary-of-deliberations-february-11-2026/", 2015) is None
    nav = "<html><body><main><p>" + "Find the latest updates and Bank communications, answers. " * 30 + "</p></main></body></html>"
    assert parse_page(nav, f"{b}/2024/06/fad-press-release-2024-06-05/", "announcement") is None
