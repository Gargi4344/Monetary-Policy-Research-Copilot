"use client";

import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceLine, Scatter, XAxis, YAxis } from "recharts";

import { BlockSkeleton, EmptyState, ErrorState, Kpi, KpiSkeleton } from "@/components/blocks";
import { Page, PageHeader } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { getJson, type ToneDetail, type ToneIndex, type ToneRow } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/format";

const config = {
  tone: { label: "Tone (hawkish − dovish share)", color: "var(--chart-2)" },
  rate: { label: "Policy rate (%)", color: "var(--chart-1)" },
} satisfies ChartConfig;

function Detail({ row }: { row: ToneRow }) {
  const q = useQuery({ queryKey: ["tone", row.doc_id], queryFn: () => getJson<ToneDetail>(`/api/tone-index/${row.doc_id}`) });
  return (
    <Card className="rounded-2xl">
      <CardHeader>
        <CardTitle className="text-base">{row.title}</CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-2">
          {formatDate(row.date)} · tone {row.tone.toFixed(2)}
          <Badge variant={row.decision === "hike" ? "destructive" : "secondary"}>{row.decision}</Badge>
          <a href={row.url} target="_blank" rel="noreferrer" className="text-teal inline-flex items-center gap-1 hover:underline">
            Read it <ExternalLink className="size-3" aria-hidden />
          </a>
        </CardDescription>
      </CardHeader>
      <CardContent>
        {q.isPending ? (
          <BlockSkeleton className="h-40" />
        ) : q.error ? (
          <ErrorState error={q.error} />
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {(["hawkish", "dovish"] as const).map((k) => (
              <section key={k}>
                <h3 className={`mb-2 text-sm font-semibold ${k === "hawkish" ? "text-hawk" : "text-dove"}`}>
                  Most {k} sentences ({q.data[k].length})
                </h3>
                <ul className="space-y-2 text-sm">
                  {q.data[k].slice(0, 4).map((s) => (
                    <li key={s.idx} className="bg-muted/50 rounded-xl p-2">
                      {s.sentence}
                      {s.finbert && s.finbert !== s.llm ? (
                        <span className="text-muted-foreground block text-xs">FinBERT: {s.finbert}</span>
                      ) : null}
                    </li>
                  ))}
                  {q.data[k].length === 0 ? <li className="text-muted-foreground">None.</li> : null}
                </ul>
              </section>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function TonePage() {
  const q = useQuery({ queryKey: ["tone-index"], queryFn: () => getJson<ToneIndex>("/api/tone-index") });
  const [selected, setSelected] = useState<ToneRow | null>(null);
  const anns = useMemo(() => q.data?.announcements ?? [], [q.data]);
  const data = useMemo(() => {
    const rate = new Map((q.data?.policy_rate ?? []).map((r) => [r.date.slice(0, 7), r.value]));
    const byMonth = new Map<string, { month: string; rate?: number; tone?: number; hike?: number; cut?: number; row?: ToneRow }>();
    for (const [m, v] of rate) byMonth.set(m, { month: m, rate: v });
    for (const a of anns) {
      const m = a.date.slice(0, 7);
      const e = byMonth.get(m) ?? { month: m };
      e.tone = a.tone;
      e.row = a;
      if (a.decision === "hike") e.hike = a.tone;
      if (a.decision === "cut") e.cut = a.tone;
      byMonth.set(m, e);
    }
    return [...byMonth.values()].sort((a, b) => a.month.localeCompare(b.month));
  }, [q.data, anns]);
  const latest = anns[anns.length - 1];
  const counts = { hike: anns.filter((a) => a.decision === "hike").length, cut: anns.filter((a) => a.decision === "cut").length };

  return (
    <>
      <PageHeader title="Tone Index" description="Hawkish minus dovish share of each rate announcement, against the policy rate" />
      <Page>
        {q.isPending ? (
          <KpiSkeleton />
        ) : q.error ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        ) : anns.length === 0 ? (
          <Card className="rounded-2xl">
            <CardContent>
              <EmptyState title="No tone index yet" message="Run `make tone` to label the announcements with the local model." />
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Kpi label="Announcements scored" value={formatNumber(anns.length)} detail={`${formatDate(anns[0].date)} to ${formatDate(latest.date)}`} />
              <Kpi label="Latest tone" value={latest.tone.toFixed(2)} detail={`${formatDate(latest.date)}: ${latest.decision}`} tone={latest.tone > 0 ? "alert" : "teal"} />
              <Kpi label="Hikes" value={formatNumber(counts.hike)} detail="Decisions from the Valet policy rate" tone="alert" />
              <Kpi label="Cuts" value={formatNumber(counts.cut)} detail="Decisions from the Valet policy rate" tone="teal" />
            </div>
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle>Tone and the policy rate</CardTitle>
                <CardDescription>
                  Tone = (hawkish − dovish) / sentences, labelled by the LLM. Red dots mark hikes, teal dots cuts. Click a point to read its most hawkish and dovish sentences.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={config} className="h-80 w-full">
                  <ComposedChart
                    data={data}
                    margin={{ left: 4, right: 4, top: 8 }}
                    onClick={(e) => {
                      const idx = e?.activeTooltipIndex;
                      const row = typeof idx === "number" ? data[idx]?.row : undefined;
                      if (row) setSelected(row);
                    }}
                  >
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="month" tickLine={false} axisLine={false} minTickGap={40} tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="tone" domain={[-1, 1]} tickLine={false} axisLine={false} width={36} tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="rate" orientation="right" tickLine={false} axisLine={false} width={36} tick={{ fontSize: 11 }} unit="%" />
                    <ReferenceLine yAxisId="tone" y={0} stroke="var(--border)" />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Line yAxisId="rate" dataKey="rate" type="stepAfter" stroke="var(--color-rate)" strokeWidth={2} dot={false} connectNulls />
                    <Line yAxisId="tone" dataKey="tone" type="monotone" stroke="var(--color-tone)" strokeWidth={2} dot={{ r: 2 }} connectNulls />
                    <Scatter yAxisId="tone" dataKey="hike" fill="var(--hawk)" name="Hike" />
                    <Scatter yAxisId="tone" dataKey="cut" fill="var(--dove)" name="Cut" />
                  </ComposedChart>
                </ChartContainer>
              </CardContent>
            </Card>
            {selected ? (
              <Detail row={selected} />
            ) : (
              <p className="text-muted-foreground text-sm">Select an announcement on the chart to see its sentences.</p>
            )}
          </>
        )}
      </Page>
    </>
  );
}
