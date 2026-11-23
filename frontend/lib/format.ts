const LOCALE = "en-CA";

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(`${iso.slice(0, 10)}T12:00:00`);
  return d.toLocaleDateString(LOCALE, { year: "numeric", month: "short", day: "numeric" });
}

export function formatPct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function formatNumber(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : v.toLocaleString(LOCALE);
}

export function formatSeconds(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : `${v.toFixed(1)} s`;
}
