"use client";

import { Page, PageHeader, useHealth } from "@/components/shell";
import { Kpi } from "@/components/blocks";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/format";

const STAGES = [
  ["Bank of Canada publications", "Scrape (robots.txt, delay, cache) → chunk by heading, ~500 tokens, 15% overlap, title + date header"],
  ["Index", "bge-small embeddings in Qdrant (embedded) + BM25"],
  ["Valet API", "Policy rate, CORRA, bond yields, USD/CAD → DuckDB"],
  ["LangGraph agent", "Router → search_documents (hybrid + RRF + reranker) → query_market_data (safe SQL) → make_chart"],
  ["Answer", "Streamed over SSE with [n] citations, the SQL shown, an inline chart and a disclaimer"],
  ["Tone index", "LLM sentence labels (rubric) + FinBERT baseline → tone score → logistic regression on the next decision"],
  ["Evaluation", "30 questions: hit rate@5, MRR, LLM-judged faithfulness, citation coverage, p50/p95 latency, guardrails"],
];

export default function AboutPage() {
  const health = useHealth();
  return (
    <>
      <PageHeader title="About" description="How the copilot works, its limits and its guardrails" />
      <Page>
        <div className="grid gap-4 sm:grid-cols-3">
          <Kpi label="Model" value={health.data?.model.split("/").pop() ?? "–"} detail="Through LiteLLM; local by default" />
          <Kpi label="Embeddings" value="bge-small" detail={health.data?.embed_model} />
          <Kpi label="Chunks" value={formatNumber(health.data?.chunks)} detail="Searchable passages" />
        </div>
        <Card className="rounded-2xl">
          <CardHeader>
            <CardTitle>Architecture</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3">
              {STAGES.map(([name, what], i) => (
                <li key={name} className="flex gap-3">
                  <span className="bg-primary text-primary-foreground grid size-7 shrink-0 place-items-center rounded-full text-xs font-semibold">{i + 1}</span>
                  <div>
                    <p className="font-medium">{name}</p>
                    <p className="text-muted-foreground text-sm">{what}</p>
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
        <div className="grid gap-4 md:grid-cols-2">
          <Card className="rounded-2xl">
            <CardHeader>
              <CardTitle>Responsible AI and guardrails</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm leading-6">
              <p>Retrieved passages are wrapped as data; the model is told never to follow instructions inside them, and tag-like text is neutralised (prompt-injection defence).</p>
              <p>The copilot refuses investment advice and off-topic questions, and says &ldquo;not found in the sources&rdquo; instead of guessing.</p>
              <p>SQL is parsed and must be a single SELECT over allowlisted tables, gets an automatic LIMIT, runs on a read-only connection, and is shown to you.</p>
              <p>Every question, tool call and latency is logged as structured JSON.</p>
            </CardContent>
          </Card>
          <Card className="rounded-2xl">
            <CardHeader>
              <CardTitle>Limitations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm leading-6">
              <p>Only the documents in the Sources Library are searched (2015 onward; HTML Monetary Policy Reports from July 2024).</p>
              <p>The evaluation questions and expected sources were drafted from the documents and still need a person to review them.</p>
              <p>Faithfulness is judged by the same local model, which tends to be lenient. Tone labels come from a small local model; FinBERT measures sentiment, not policy stance.</p>
              <p>With a few dozen decisions, the tone model&apos;s accuracy is a noisy estimate; compare it with the &ldquo;same as last time&rdquo; baseline in RESULTS.md.</p>
              <p>This is a research aid, not investment advice.</p>
            </CardContent>
          </Card>
        </div>
      </Page>
    </>
  );
}
