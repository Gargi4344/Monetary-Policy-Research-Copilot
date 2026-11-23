// Server-sent events over POST (EventSource only supports GET).

import { API_URL } from "@/lib/api";

export interface SseEvent {
  event: string;
  data: unknown;
}

/** Parse complete events out of a buffer; returns the events and the unparsed rest. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() ?? "";
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length) continue;
    const raw = data.join("\n");
    try {
      events.push({ event, data: JSON.parse(raw) });
    } catch {
      events.push({ event, data: raw });
    }
  }
  return { events, rest };
}

export async function streamChat(
  question: string,
  onEvent: (e: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name !== "AbortError")
      onEvent({ event: "error", data: { message: "The API is not reachable. Is `make api` running?" } });
    return;
  }
  if (!res.ok || !res.body) {
    onEvent({ event: "error", data: { message: `The API returned ${res.status}.` } });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
}

/** Split an answer into text and citation markers ([1], [SQL]) for rendering chips. */
export function splitCitations(answer: string): (string | number | "SQL")[] {
  const parts: (string | number | "SQL")[] = [];
  let last = 0;
  for (const m of answer.matchAll(/\[(\d{1,2}|SQL)\]/g)) {
    if (m.index > last) parts.push(answer.slice(last, m.index));
    parts.push(m[1] === "SQL" ? "SQL" : Number(m[1]));
    last = m.index + m[0].length;
  }
  if (last < answer.length) parts.push(answer.slice(last));
  return parts;
}
