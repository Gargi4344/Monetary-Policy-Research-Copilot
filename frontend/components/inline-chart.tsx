"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import { type ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import type { ChartSpec } from "@/lib/api";

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-4)", "var(--chart-3)", "var(--chart-5)"];

/** Renders the agent's chart spec (built from SQL rows) as a Recharts line or bar chart. */
export function InlineChart({ spec }: { spec: ChartSpec }) {
  const config = Object.fromEntries(
    spec.series.map((s, i) => [s, { label: s.replace(/_/g, " "), color: COLORS[i % COLORS.length] }]),
  ) satisfies ChartConfig;
  const data = spec.data.map((d) => ({ ...d, [spec.x]: String(d[spec.x]).slice(0, 10) }));
  const Chart = spec.type === "line" ? LineChart : BarChart;
  return (
    <figure className="space-y-1">
      <ChartContainer config={config} className="h-64 w-full">
        <Chart data={data} margin={{ left: 4, right: 12, top: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey={spec.x} tickLine={false} axisLine={false} minTickGap={32} tick={{ fontSize: 11 }} />
          <YAxis tickLine={false} axisLine={false} width={40} tick={{ fontSize: 11 }} />
          <ChartTooltip content={<ChartTooltipContent />} />
          {spec.series.length > 1 ? <ChartLegend content={<ChartLegendContent />} /> : null}
          {spec.series.map((s) =>
            spec.type === "line" ? (
              <Line key={s} dataKey={s} type="monotone" stroke={`var(--color-${s})`} strokeWidth={2} dot={false} connectNulls />
            ) : (
              <Bar key={s} dataKey={s} fill={`var(--color-${s})`} radius={4} />
            ),
          )}
        </Chart>
      </ChartContainer>
      <figcaption className="text-muted-foreground text-xs">
        {spec.series.join(", ").replace(/_/g, " ")} by {spec.x} ({spec.data.length} points, from the SQL result)
      </figcaption>
    </figure>
  );
}
