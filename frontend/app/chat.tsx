"use client";

import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, CircleHelp, Code2, ExternalLink, Send, ShieldAlert, Square, Workflow } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Kpi, KpiSkeleton } from "@/components/blocks";
import { InlineChart } from "@/components/inline-chart";
import { Page, PageHeader, useHealth } from "@/components/shell";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import {
  type ChartSpec,
  type Citation,
  DOC_TYPE_LABEL,
  type Done,
  type EvalResult,
  getJson,
  type Source,
  type SqlPayload,
  type Step,
} from "@/lib/api";
import { formatDate, formatNumber, formatPct } from "@/lib/format";
import { splitCitations, streamChat } from "@/lib/sse";

const EXAMPLES = [
  "How did the Bank explain the 2022 rate hikes?",
  "Show the policy rate against the 10-year yield since 2020",
  "Why did the Bank cut its policy rate in June 2024?",
  "What did the Governing Council say about tariffs in 2025?",
];

interface Turn {
  question: string;
  steps: Step[];
  sources: Source[];
  sql: SqlPayload | null;
  chart: ChartSpec | null;
  answer: string;
  done: Done | null;
  error: string | null;
}

const STEP_TEXT: Record<Step["tool"], (s: Step) => string> = {
  route: (s) => (s.category === "in_scope" ? `Planned: ${s.tools?.join(", ").replace(/_/g, " ") || "answer"}` : `Refused: ${s.category?.replace(/_/g, " ")}`),
  search_documents: (s) => `Searched the documents for "${s.query}" → ${s.results} passages`,
  query_market_data: (s) => (s.error ? `SQL failed: ${s.error}` : `Ran SQL → ${s.rows} rows`),
  make_chart: (s) => `Built a ${s.type ?? ""} chart`,
};

function Trace({ steps }: { steps: Step[] }) {
  const [open, setOpen] = useState(false);
  if (!steps.length) return null;
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs">
        <Workflow className="size-3.5" aria-hidden /> How I answered
        <ChevronDown className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ol className="border-teal/40 mt-2 space-y-1 border-l-2 pl-3 text-xs">
          {steps.map((s, i) => (
            <li key={i}>
              {STEP_TEXT[s.tool](s)}
              {s.ms !== undefined ? <span className="text-muted-foreground"> · {s.ms} ms</span> : null}
            </li>
          ))}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
}

function AnswerText({ turn, onCite }: { turn: Turn; onCite: (s: Source) => void }) {
  const byN = new Map(turn.sources.map((s) => [s.n, s]));
  return (
    <p className="text-[15px] leading-7 whitespace-pre-wrap">
      {splitCitations(turn.answer).map((p, i) => {
        if (p === "SQL")
          return (
            <Badge key={i} variant="secondary" className="mx-0.5 align-middle font-mono text-[10px]">
              SQL
            </Badge>
          );
        if (typeof p === "string") return <span key={i}>{p}</span>;
        const src = byN.get(p);
        return src ? (
          <button
            key={i}
            type="button"
            onClick={() => onCite(src)}
            className="bg-teal/15 text-teal hover:bg-teal/25 mx-0.5 rounded-md px-1.5 py-0.5 align-middle text-xs font-semibold"
            title={`${src.title} (${formatDate(src.date)})`}
          >
            [{p}]<span className="sr-only">: {src.title}, {formatDate(src.date)}</span>
          </button>
        ) : (
          <span key={i}>[{p}]</span>
        );
      })}
      {!turn.done && !turn.error ? <span className="bg-foreground ml-0.5 inline-block h-4 w-1.5 animate-pulse align-middle" aria-hidden /> : null}
    </p>
  );
}

function TurnCard({ turn, onCite }: { turn: Turn; onCite: (s: Source) => void }) {
  const refused = turn.done?.refused;
  const notFound = turn.done && !turn.done.found && !refused;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <Card className="rounded-2xl">
        <CardHeader>
          <CardTitle className="text-base">{turn.question}</CardTitle>
          <Trace steps={turn.steps} />
        </CardHeader>
        <CardContent className="space-y-4">
          {turn.error ? (
            <Alert variant="destructive">
              <AlertDescription>{turn.error}</AlertDescription>
            </Alert>
          ) : refused ? (
            <div className="flex gap-2 text-sm">
              <ShieldAlert className="text-alert mt-0.5 size-4 shrink-0" aria-hidden />
              <p>{turn.done?.answer}</p>
            </div>
          ) : notFound ? (
            <div className="text-muted-foreground flex gap-2 text-sm">
              <CircleHelp className="mt-0.5 size-4 shrink-0" aria-hidden />
              <p>{turn.done?.answer}</p>
            </div>
          ) : turn.answer ? (
            <AnswerText turn={turn} onCite={onCite} />
          ) : (
            <p className="text-muted-foreground text-sm">{turn.steps.length ? "Writing the answer…" : "Planning…"}</p>
          )}
          {turn.chart ? <InlineChart spec={turn.chart} /> : null}
          {turn.sql?.sql ? (
            <details className="group">
              <summary className="text-muted-foreground inline-flex cursor-pointer items-center gap-1 text-xs">
                <Code2 className="size-3.5" aria-hidden /> SQL that ran ({formatNumber(turn.sql.rows?.length)} rows)
              </summary>
              <pre className="bg-muted mt-2 overflow-x-auto rounded-xl p-3 font-mono text-xs">{turn.sql.sql}</pre>
            </details>
          ) : null}
          {turn.done?.found && turn.done.citations.length ? (
            <div className="space-y-1">
              <p className="text-muted-foreground text-xs font-medium">Sources cited</p>
              <ul className="flex flex-wrap gap-2">
                {turn.done.citations.map((c: Citation) => (
                  <li key={c.n}>
                    <button
                      type="button"
                      onClick={() => onCite(turn.sources.find((s) => s.n === c.n)!)}
                      className="hover:border-teal rounded-xl border px-2.5 py-1 text-left text-xs"
                    >
                      <span className="text-teal font-semibold">[{c.n}]</span> {c.title.slice(0, 60)} · {formatDate(c.date)}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {turn.done && !refused ? (
            <p className="text-muted-foreground text-xs">
              {turn.done.disclaimer} · {turn.done.latency_s.toFixed(1)} s · {turn.done.model}
            </p>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ChatPage() {
  const health = useHealth();
  const evalRun = useQuery({ queryKey: ["eval"], queryFn: () => getJson<EvalResult>("/api/eval/latest"), retry: false });
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<Source | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns]);

  const update = (fn: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), fn(ts[ts.length - 1])]);

  async function ask(q: string) {
    const text = q.trim();
    if (text.length < 3 || busy) return;
    setQuestion("");
    setBusy(true);
    setTurns((ts) => [...ts, { question: text, steps: [], sources: [], sql: null, chart: null, answer: "", done: null, error: null }]);
    abort.current = new AbortController();
    await streamChat(
      text,
      ({ event, data }) => {
        if (event === "step") update((t) => ({ ...t, steps: [...t.steps, data as Step] }));
        else if (event === "sources") update((t) => ({ ...t, sources: data as Source[] }));
        else if (event === "sql") update((t) => ({ ...t, sql: data as SqlPayload }));
        else if (event === "chart") update((t) => ({ ...t, chart: data as ChartSpec }));
        else if (event === "token") update((t) => ({ ...t, answer: t.answer + String(data) }));
        else if (event === "done") update((t) => ({ ...t, done: data as Done, answer: (data as Done).answer }));
        else if (event === "error") update((t) => ({ ...t, error: (data as { message: string }).message }));
      },
      abort.current.signal,
    );
    update((t) => (t.done || t.error ? t : { ...t, error: "Stopped." }));
    setBusy(false);
  }

  const m = evalRun.data?.metrics;
  return (
    <>
      <PageHeader title="Chat" description="Ask about Canadian monetary policy. Every claim is cited." />
      <Page>
        {health.isPending ? (
          <KpiSkeleton />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Kpi label="Documents indexed" value={formatNumber(health.data?.documents)} detail="Bank of Canada publications" />
            <Kpi label="Chunks searchable" value={formatNumber(health.data?.chunks)} detail="Hybrid: vectors + BM25, reranked" />
            <Kpi label="Retrieval hit rate@5" value={formatPct(m?.hit_rate_at_5)} detail={m ? `${m.questions}-question evaluation` : "Run make eval"} tone="teal" />
            <Kpi label="Answer faithfulness" value={formatPct(m?.faithfulness)} detail={m ? "LLM-as-judge" : "Run make eval"} />
          </div>
        )}
        {health.error ? (
          <Alert variant="destructive">
            <AlertDescription>{(health.error as Error).message}</AlertDescription>
          </Alert>
        ) : null}

        <section aria-label="Conversation" className="space-y-4">
          {turns.length === 0 ? (
            <Card className="rounded-2xl">
              <CardContent className="space-y-3 py-6">
                <p className="text-sm">Try a question. The copilot searches the publications, queries market data with read-only SQL when numbers are needed, and cites every source.</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLES.map((e) => (
                    <Button key={e} variant="outline" size="sm" className="h-auto rounded-xl py-1.5 text-left whitespace-normal" onClick={() => ask(e)}>
                      {e}
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : null}
          <div aria-live="polite" className="space-y-4">
            <AnimatePresence>
              {turns.map((t, i) => (
                <TurnCard key={i} turn={t} onCite={setOpen} />
              ))}
            </AnimatePresence>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
            className="bg-background sticky bottom-0 space-y-2 border-t pt-3 pb-2"
          >
            <label htmlFor="question" className="sr-only">
              Your question
            </label>
            <Textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(question);
                }
              }}
              placeholder="Ask about rate decisions, inflation, the outlook… (Enter to send)"
              rows={2}
              maxLength={1000}
              className="rounded-2xl"
            />
            <div className="flex justify-end gap-2">
              {busy ? (
                <Button type="button" variant="outline" onClick={() => abort.current?.abort()}>
                  <Square className="size-4" aria-hidden /> Stop
                </Button>
              ) : null}
              <Button type="submit" disabled={busy || question.trim().length < 3} className="rounded-xl">
                <Send className="size-4" aria-hidden /> Ask
              </Button>
            </div>
          </form>
          <div ref={bottom} />
        </section>
      </Page>

      <Sheet open={open !== null} onOpenChange={(o) => !o && setOpen(null)}>
        <SheetContent className="w-full sm:max-w-lg">
          {open ? (
            <>
              <SheetHeader>
                <SheetTitle>
                  [{open.n}] {open.title}
                </SheetTitle>
                <SheetDescription>
                  {DOC_TYPE_LABEL[open.doc_type]} · {formatDate(open.date)}
                  {open.section ? ` · ${open.section}` : ""}
                </SheetDescription>
              </SheetHeader>
              <div className="space-y-3 overflow-y-auto px-4 pb-6">
                <p className="text-muted-foreground text-xs">The passage the answer was built from:</p>
                <blockquote className="border-teal bg-muted/50 rounded-r-xl border-l-4 p-3 text-sm leading-6 whitespace-pre-wrap">
                  <mark className="passage">{open.text}</mark>
                </blockquote>
                <a href={open.url} target="_blank" rel="noreferrer" className="text-teal inline-flex items-center gap-1 text-sm hover:underline">
                  Open on bankofcanada.ca <ExternalLink className="size-3.5" aria-hidden />
                </a>
              </div>
            </>
          ) : null}
        </SheetContent>
      </Sheet>
    </>
  );
}
