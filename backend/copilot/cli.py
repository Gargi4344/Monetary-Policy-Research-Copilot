"""`copilot` command line: ingest, index, tone, eval, results."""

from __future__ import annotations

import json

import typer

from copilot.config import get_settings
from copilot.logs import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)


_started = 0.0


@app.callback()
def main(ctx: typer.Context) -> None:
    """Configure JSON logging and time every command (data/runlog.json)."""
    import time

    global _started
    configure_logging(get_settings().log_level)
    _started = time.perf_counter()
    if ctx.invoked_subcommand in {"scrape", "market", "chunk", "index", "tone", "eval"}:
        from copilot.results import record_step

        ctx.call_on_close(
            lambda: record_step(ctx.invoked_subcommand, time.perf_counter() - _started)
        )


@app.command()
def scrape(limit: int | None = None) -> None:
    """Collect Bank of Canada publications (polite, cached) into data/docs.jsonl."""
    from collections import Counter

    from copilot.ingest.scrape import scrape as run

    docs = run(limit)
    typer.echo(json.dumps({"documents": len(docs), "by_type": Counter(d.doc_type for d in docs)}))


@app.command()
def market() -> None:
    """Load Valet API series into DuckDB."""
    from copilot.ingest.valet import ingest_market_data

    typer.echo(json.dumps(ingest_market_data()))


@app.command()
def chunk() -> None:
    """Chunk data/docs.jsonl into data/chunks.jsonl."""
    from copilot.ingest.chunk import chunk_all
    from copilot.ingest.scrape import load_docs

    chunks = chunk_all(load_docs())
    typer.echo(json.dumps({"chunks": len(chunks)}))


@app.command()
def index() -> None:
    """Embed chunks into Qdrant and build the BM25 index."""
    from copilot.index.store import build_indexes
    from copilot.ingest.chunk import load_chunks

    typer.echo(json.dumps(build_indexes(load_chunks())))


@app.command()
def search(q: str, k: int = 5) -> None:
    """Try the hybrid search from the command line."""
    from copilot.index.retrieve import search as run

    for h in run(q, top_k=k):
        typer.echo(f"{h.rerank_score:>7} {h.date} {h.doc_type:<17} {h.title[:70]}")


@app.command()
def ask(question: str) -> None:
    """Ask the copilot from the command line."""
    from copilot.agent.graph import answer_stream, run_agent

    state = run_agent(question)
    for step in state.get("steps", []):
        typer.echo(f"  > {json.dumps(step, default=str)[:300]}")
    for tok in answer_stream(state):
        typer.echo(tok, nl=False)
    typer.echo()


@app.command()
def tone(finbert: bool = True) -> None:
    """Label announcement sentences, build the tone index and test next-decision prediction."""
    from dataclasses import asdict

    from copilot.ingest.scrape import load_docs
    from copilot.tone.index import build_tone_index

    run = build_tone_index(load_docs(), use_finbert=finbert)
    out = get_settings().data_dir / "eval" / "tone.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(run), indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps(asdict(run), indent=2, default=str))


@app.command("tone-sample")
def tone_sample(n: int = 50) -> None:
    """Write n sentences to data/labels/human_tone.csv for hand-labelling."""
    from copilot.tone.human import export_sample

    typer.echo(f"wrote {export_sample(n)}; fill human_label, then run `copilot tone-agreement`")


@app.command("tone-agreement")
def tone_agreement() -> None:
    """Accuracy of LLM and FinBERT labels against your hand labels."""
    from copilot.tone.human import agreement

    typer.echo(json.dumps(agreement(), indent=2))


@app.command("eval")
def eval_cmd(limit: int | None = None) -> None:
    """Run the 30-question evaluation and the guardrail suite."""
    from copilot.evals.harness import run_eval

    result = run_eval(limit=limit)
    typer.echo(
        json.dumps(
            {
                "metrics": result["metrics"],
                "guardrails": {
                    k: v for k, v in result["guardrails"].items() if k != "prompt_checks"
                },
            },
            indent=2,
        )
    )


@app.command()
def results() -> None:
    """Regenerate RESULTS.md and the README results and highlights from this run's outputs."""
    from copilot.results import write_results

    typer.echo(write_results())


if __name__ == "__main__":
    app()
