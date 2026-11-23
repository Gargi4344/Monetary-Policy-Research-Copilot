"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BookOpenText, Database, FlaskConical, Info, LineChart, Landmark, MessagesSquare } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { getJson, type Health, type SourcesResponse } from "@/lib/api";
import { formatNumber } from "@/lib/format";

const NAV = [
  { href: "/", title: "Chat", icon: MessagesSquare },
  { href: "/tone", title: "Tone Index", icon: LineChart },
  { href: "/evaluation", title: "Evaluation", icon: FlaskConical },
  { href: "/sources", title: "Sources Library", icon: BookOpenText },
  { href: "/about", title: "About", icon: Info },
];

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: () => getJson<Health>("/api/health"), staleTime: 30_000 });
}

/** "About this data" drawer: sources, licences and when the data was last refreshed. */
export function AboutDataDrawer() {
  const sources = useQuery({
    queryKey: ["sources-meta"],
    queryFn: () => getJson<SourcesResponse>("/api/sources"),
  });
  const health = useHealth();
  return (
    <Sheet>
      <SheetTrigger render={<Button variant="outline" size="sm" className="rounded-xl" />}>
        <Database className="size-4" aria-hidden /> About this data
      </SheetTrigger>
      <SheetContent className="w-full sm:max-w-md">
        <SheetHeader>
          <SheetTitle>About this data</SheetTitle>
          <SheetDescription>Where every number and passage comes from.</SheetDescription>
        </SheetHeader>
        <div className="space-y-5 px-4 pb-6 text-sm leading-6">
          <section>
            <h3 className="font-semibold">Bank of Canada publications</h3>
            <p className="text-muted-foreground">
              Interest rate announcements, opening statements, summaries of deliberations and Monetary Policy
              Report overviews from bankofcanada.ca (2015 onward), collected politely from the public sitemap.
              Used under the{" "}
              <a className="underline" href="https://www.bankofcanada.ca/terms/" target="_blank" rel="noreferrer">
                Bank of Canada terms of use
              </a>
              .
            </p>
          </section>
          <section>
            <h3 className="font-semibold">Market data</h3>
            <p className="text-muted-foreground">
              Policy rate, CORRA, 2/5/10-year Government of Canada benchmark yields and USD/CAD from the Bank of
              Canada Valet API. Statistics Canada data, where used, is under the Open Government Licence – Canada.
            </p>
          </section>
          <dl className="bg-muted/60 grid grid-cols-2 gap-3 rounded-2xl p-4">
            <div>
              <dt className="text-muted-foreground text-xs">Documents</dt>
              <dd className="font-semibold">{formatNumber(sources.data?.total)}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground text-xs">Chunks indexed</dt>
              <dd className="font-semibold">{formatNumber(health.data?.chunks)}</dd>
            </div>
            <div className="col-span-2">
              <dt className="text-muted-foreground text-xs">Last refreshed</dt>
              <dd className="font-semibold">{sources.data?.last_refreshed ?? "not yet – run make data"}</dd>
            </div>
            <div className="col-span-2">
              <dt className="text-muted-foreground text-xs">Model</dt>
              <dd className="font-semibold break-all">{health.data?.model ?? "–"}</dd>
            </div>
          </dl>
          <p className="text-muted-foreground text-xs">
            AI-generated answers can be wrong. Every answer cites its sources: verify them before relying on it.
          </p>
        </div>
      </SheetContent>
    </Sheet>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <SidebarProvider>
      <Sidebar collapsible="icon">
        <SidebarHeader>
          <Link href="/" className="flex items-center gap-2 rounded-xl px-2 py-2">
            <span className="bg-primary text-primary-foreground grid size-8 shrink-0 place-items-center rounded-xl">
              <Landmark className="size-4" aria-hidden />
            </span>
            <span className="flex flex-col leading-tight group-data-[collapsible=icon]:hidden">
              <span className="text-sm font-semibold">Policy Research Copilot</span>
              <span className="text-muted-foreground text-xs">Bank of Canada sources</span>
            </span>
          </Link>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV.map((item) => (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      isActive={item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href)}
                      tooltip={item.title}
                      render={<Link href={item.href} />}
                    >
                      <item.icon aria-hidden />
                      <span>{item.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <p className="text-muted-foreground px-2 text-xs group-data-[collapsible=icon]:hidden">
            Not investment advice. Answers are AI-generated: verify the sources.
          </p>
        </SidebarFooter>
      </Sidebar>
      <SidebarInset>{children}</SidebarInset>
    </SidebarProvider>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: React.ReactNode }) {
  return (
    <header className="bg-background/90 sticky top-0 z-20 border-b backdrop-blur">
      <div className="flex flex-wrap items-center gap-3 px-4 py-3 md:px-8">
        <SidebarTrigger className="-ml-1" aria-label="Toggle navigation" />
        <div className="mr-auto min-w-0">
          <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">Monetary Policy Research Copilot</p>
          <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>
          {description ? <p className="text-muted-foreground hidden text-sm sm:block">{description}</p> : null}
        </div>
        <div className="flex items-center gap-2">
          {actions}
          <AboutDataDrawer />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export function Page({ children }: { children: React.ReactNode }) {
  return (
    <motion.main
      id="main"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 md:px-8"
    >
      {children}
    </motion.main>
  );
}
