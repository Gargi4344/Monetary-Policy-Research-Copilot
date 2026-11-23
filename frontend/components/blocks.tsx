"use client";

import { AlertTriangle, Inbox, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function Kpi({
  label,
  value,
  detail,
  tone = "default",
}: {
  label: string;
  value: string;
  detail?: React.ReactNode;
  tone?: "default" | "teal" | "alert";
}) {
  return (
    <Card className="rounded-2xl">
      <CardContent className="space-y-1">
        <p className="text-muted-foreground text-sm font-medium">{label}</p>
        <p
          className={cn(
            "text-3xl font-semibold tracking-tight tabular-nums",
            tone === "teal" && "text-teal",
            tone === "alert" && "text-alert",
          )}
        >
          {value}
        </p>
        {detail ? <p className="text-muted-foreground text-xs">{detail}</p> : null}
      </CardContent>
    </Card>
  );
}

export function KpiSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-busy="true" aria-label="Loading">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className="h-28 rounded-2xl" />
      ))}
    </div>
  );
}

export function BlockSkeleton({ className = "h-80" }: { className?: string }) {
  return <Skeleton className={cn("w-full rounded-2xl", className)} aria-busy="true" aria-label="Loading" />;
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <Card role="alert" className="rounded-2xl">
      <CardContent className="flex flex-col items-center gap-3 py-10 text-center">
        <AlertTriangle className="text-alert size-8" aria-hidden />
        <div>
          <p className="font-medium">We couldn&apos;t load this</p>
          <p className="text-muted-foreground mt-1 max-w-md text-sm">
            {error instanceof Error ? error.message : "Something went wrong."}
          </p>
        </div>
        {onRetry ? (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="size-4" aria-hidden /> Try again
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function EmptyState({ title, message }: { title: string; message: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 py-12 text-center">
      <Inbox className="text-muted-foreground size-8" aria-hidden />
      <p className="font-medium">{title}</p>
      <p className="text-muted-foreground max-w-md text-sm">{message}</p>
    </div>
  );
}
