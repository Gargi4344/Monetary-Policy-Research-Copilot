"use client";

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { BlockSkeleton, EmptyState, ErrorState, Kpi, KpiSkeleton } from "@/components/blocks";
import { Page, PageHeader } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { DOC_TYPE_LABEL, getJson, type SourcesResponse } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/format";

const TYPES = ["all", "announcement", "opening_statement", "deliberations", "mpr"] as const;

export default function SourcesPage() {
  const q = useQuery({ queryKey: ["sources"], queryFn: () => getJson<SourcesResponse>("/api/sources") });
  const [text, setText] = useState("");
  const [type, setType] = useState<(typeof TYPES)[number]>("all");
  const docs = useMemo(
    () =>
      (q.data?.documents ?? []).filter(
        (d) => (type === "all" || d.doc_type === type) && (!text || d.title.toLowerCase().includes(text.toLowerCase()) || d.date.includes(text)),
      ),
    [q.data, text, type],
  );
  const all = q.data?.documents ?? [];
  const count = (t: string) => all.filter((d) => d.doc_type === t).length;
  return (
    <>
      <PageHeader title="Sources Library" description="Every indexed Bank of Canada document" />
      <Page>
        {q.isPending ? (
          <KpiSkeleton />
        ) : q.error ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Kpi label="Rate announcements" value={formatNumber(count("announcement"))} />
            <Kpi label="Opening statements" value={formatNumber(count("opening_statement"))} />
            <Kpi label="Deliberation summaries" value={formatNumber(count("deliberations"))} />
            <Kpi label="MPR overviews" value={formatNumber(count("mpr"))} />
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="text-muted-foreground absolute top-2.5 left-2.5 size-4" aria-hidden />
            <label htmlFor="src-search" className="sr-only">
              Search titles or dates
            </label>
            <Input id="src-search" value={text} onChange={(e) => setText(e.target.value)} placeholder="Search titles or dates (2024-06)" className="w-72 rounded-xl pl-8" />
          </div>
          <ToggleGroup value={[type]} onValueChange={(v) => v[0] && setType(v[0] as (typeof TYPES)[number])} aria-label="Document type">
            {TYPES.map((t) => (
              <ToggleGroupItem key={t} value={t} className="rounded-xl px-3 text-xs">
                {t === "all" ? "All" : DOC_TYPE_LABEL[t]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        {q.isPending ? (
          <BlockSkeleton />
        ) : docs.length === 0 ? (
          <Card className="rounded-2xl">
            <CardContent>
              <EmptyState title="No documents" message={all.length ? "Nothing matches these filters." : "Run `make data` to collect the publications."} />
            </CardContent>
          </Card>
        ) : (
          <Card className="rounded-2xl">
            <CardContent>
              <p className="text-muted-foreground mb-3 text-sm">{formatNumber(docs.length)} documents</p>
              <div className="overflow-x-auto rounded-xl border">
                <Table>
                  <caption className="sr-only">Indexed documents</caption>
                  <TableHeader className="bg-muted/50">
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Title</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead className="text-right">Words</TableHead>
                      <TableHead className="text-right">Chunks</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {docs.map((d) => (
                      <TableRow key={d.doc_id}>
                        <TableCell className="whitespace-nowrap tabular-nums">{formatDate(d.date)}</TableCell>
                        <TableCell>
                          <a href={d.url} target="_blank" rel="noreferrer" className="hover:text-teal inline-flex items-center gap-1">
                            {d.title} <ExternalLink className="size-3 opacity-60" aria-hidden />
                          </a>
                        </TableCell>
                        <TableCell>
                          <Badge variant="secondary">{DOC_TYPE_LABEL[d.doc_type]}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{formatNumber(d.words)}</TableCell>
                        <TableCell className="text-right tabular-nums">{formatNumber(d.chunks)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        )}
      </Page>
    </>
  );
}
