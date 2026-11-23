"use client";

import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";

import { BlockSkeleton, EmptyState, ErrorState, Kpi, KpiSkeleton } from "@/components/blocks";
import { Page, PageHeader } from "@/components/shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError, type EvalResult, getJson } from "@/lib/api";
import { formatPct, formatSeconds } from "@/lib/format";

function Mark({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="text-muted-foreground">–</span>;
  return ok ? (
    <CheckCircle2 className="text-teal size-4" aria-label="pass" />
  ) : (
    <XCircle className="text-alert size-4" aria-label="fail" />
  );
}

export default function EvaluationPage() {
  const q = useQuery({ queryKey: ["eval"], queryFn: () => getJson<EvalResult>("/api/eval/latest"), retry: false });
  const missing = q.error instanceof ApiError && q.error.status === 404;
  const m = q.data?.metrics;
  const g = q.data?.guardrails;
  return (
    <>
      <PageHeader title="Evaluation" description="Measured on the 30-question test set, not assumed" />
      <Page>
        {q.isPending ? (
          <>
            <KpiSkeleton />
            <BlockSkeleton />
          </>
        ) : missing ? (
          <Card className="rounded-2xl">
            <CardContent>
              <EmptyState title="No evaluation yet" message="Run `make eval`, then refresh." />
            </CardContent>
          </Card>
        ) : q.error ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Kpi label="Retrieval hit rate@5" value={formatPct(m?.hit_rate_at_5)} detail={`MRR ${m?.mrr_at_5?.toFixed(2) ?? "–"}`} tone="teal" />
              <Kpi label="Faithfulness" value={formatPct(m?.faithfulness)} detail={`LLM judge, ${m?.faithfulness_judged} answers`} />
              <Kpi label="Citation coverage" value={formatPct(m?.citation_coverage)} detail="Answer sentences with a citation" />
              <Kpi label="Latency p50 / p95" value={`${formatSeconds(m?.latency_p50_s)}`} detail={`p95 ${formatSeconds(m?.latency_p95_s)} · ${m?.model}`} />
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <Kpi label="Questions passed" value={`${m?.passed} / ${m?.questions}`} detail="Hit@5, answered, faithfulness ≥ 80%" />
              <Kpi label="Prompt guardrails" value={`${g?.prompt_passed} / ${g?.prompt_total}`} detail="Advice / off-topic refusals, injected instructions ignored" />
              <Kpi label="SQL attacks blocked" value={`${g?.sql_attacks_blocked} / ${g?.sql_attacks_total}`} detail="DROP, UPDATE, other tables, file reads…" tone="teal" />
            </div>
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Questions</CardTitle>
                <CardDescription>
                  Expected sources were drafted from the documents and are marked for human review (eval/questions.yaml).
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto rounded-xl border">
                  <Table>
                    <caption className="sr-only">Evaluation questions</caption>
                    <TableHeader className="bg-muted/50">
                      <TableRow>
                        <TableHead>Question</TableHead>
                        <TableHead>Hit@5</TableHead>
                        <TableHead className="text-right">Faithful</TableHead>
                        <TableHead className="text-right">Cited</TableHead>
                        <TableHead>Tools</TableHead>
                        <TableHead className="text-right">Time</TableHead>
                        <TableHead>Pass</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {q.data.rows.map((r) => (
                        <TableRow key={r.id}>
                          <TableCell className="max-w-md">
                            <details>
                              <summary className="cursor-pointer text-sm">{r.question}</summary>
                              <p className="text-muted-foreground mt-2 text-xs whitespace-pre-wrap">{r.answer}</p>
                            </details>
                          </TableCell>
                          <TableCell>
                            <Mark ok={r.hit_at_5} />
                          </TableCell>
                          <TableCell className="text-right tabular-nums">{formatPct(r.faithfulness, 0)}</TableCell>
                          <TableCell className="text-right tabular-nums">{formatPct(r.citation_coverage, 0)}</TableCell>
                          <TableCell className="text-xs">{r.tools.join(", ").replace(/_/g, " ")}</TableCell>
                          <TableCell className="text-right tabular-nums">{formatSeconds(r.latency_s)}</TableCell>
                          <TableCell>
                            <Mark ok={r.pass} />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Guardrail checks</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {g?.prompt_checks.map((c) => (
                    <li key={c.question} className="flex items-start gap-2">
                      <Mark ok={c.passed} />
                      <span>
                        <span className="text-muted-foreground">{c.kind === "refuse" ? "Refuse" : "Injection"}:</span> {c.question}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </>
        )}
      </Page>
    </>
  );
}
