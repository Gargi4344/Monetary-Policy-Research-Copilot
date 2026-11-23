// Typed client for the FastAPI backend.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type DocType = "announcement" | "opening_statement" | "deliberations" | "mpr";
export const DOC_TYPE_LABEL: Record<string, string> = {
  announcement: "Rate announcement",
  opening_statement: "Opening statement",
  deliberations: "Deliberations summary",
  mpr: "Monetary Policy Report",
};

export interface Source {
  n: number;
  chunk_id: string;
  doc_id: string;
  title: string;
  date: string;
  doc_type: DocType;
  url: string;
  section: string;
  text: string;
  rerank_score: number | null;
}

export interface Citation {
  n: number;
  title: string;
  date: string;
  url: string;
  doc_type: DocType;
  chunk_id: string;
  text: string;
}

export interface Step {
  tool: "route" | "search_documents" | "query_market_data" | "make_chart";
  ms?: number;
  category?: string;
  tools?: string[];
  query?: string;
  results?: number;
  sql?: string;
  rows?: number;
  error?: string | null;
  type?: string | null;
}

export interface SqlPayload {
  sql?: string;
  columns?: string[];
  rows?: (string | number | null)[][];
  truncated?: boolean;
  error?: string;
}

export interface ChartSpec {
  type: "line" | "bar";
  x: string;
  series: string[];
  data: Record<string, string | number | null>[];
}

export interface Done {
  answer: string;
  found: boolean;
  refused: boolean;
  citations: Citation[];
  latency_s: number;
  model: string;
  disclaimer: string;
}

export interface Health {
  status: string;
  model: string;
  embed_model: string;
  rerank_model: string;
  documents: number;
  chunks: number;
}

export interface SourceDoc {
  doc_id: string;
  title: string;
  date: string;
  doc_type: DocType;
  url: string;
  words: number;
  chunks: number;
}

export interface SourcesResponse {
  total: number;
  last_refreshed: string | null;
  documents: SourceDoc[];
}

export interface ToneRow {
  date: string;
  doc_id: string;
  title: string;
  url: string;
  tone: number;
  decision: "hike" | "hold" | "cut" | "unknown";
  policy_rate: number | null;
}

export interface ToneIndex {
  announcements: ToneRow[];
  policy_rate: { date: string; value: number }[];
}

export interface ToneSentence {
  idx: number;
  sentence: string;
  llm: string;
  finbert: string | null;
}

export interface ToneDetail {
  doc_id: string;
  hawkish: ToneSentence[];
  dovish: ToneSentence[];
  counts: Record<string, number>;
}

export interface EvalRow {
  id: string;
  question: string;
  expected_docs: string[];
  hit_at_5: boolean | null;
  reciprocal_rank: number | null;
  tools: string[];
  answered: boolean;
  faithfulness: number | null;
  citation_coverage: number;
  latency_s: number;
  pass: boolean;
  answer: string;
  sql: string | null;
}

export interface EvalResult {
  metrics: {
    questions: number;
    hit_rate_at_5: number | null;
    mrr_at_5: number | null;
    faithfulness: number | null;
    faithfulness_judged: number;
    citation_coverage: number | null;
    answered: number;
    passed: number;
    latency_p50_s: number;
    latency_p95_s: number;
    model: string;
    at: string;
  };
  guardrails: {
    prompt_checks: { kind: string; question: string; passed: boolean; reply?: string }[];
    prompt_passed: number;
    prompt_total: number;
    sql_attacks_blocked: number;
    sql_attacks_total: number;
  };
  rows: EvalRow[];
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | null | undefined>;

export function buildUrl(path: string, params: Params = {}): string {
  const url = new URL(`${API_URL}${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, String(v));
  }
  return url.toString();
}

export async function getJson<T>(path: string, params?: Params): Promise<T> {
  let res: Response;
  try {
    res = await fetch(buildUrl(path, params));
  } catch {
    throw new ApiError(0, "The API is not reachable. Start it with `make api` (port 8000).");
  }
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    try {
      message = (await res.json()).error?.message ?? message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}
