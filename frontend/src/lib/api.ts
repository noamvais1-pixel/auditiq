export interface Source {
  chunk_id: string;
  citation: string;
  text: string;
  score: number;
  pinned: boolean;
}

export interface Grounding {
  valid_citations: string[];
  invalid_citations: string[];
  uncited_claims: string[];
  blocked: boolean;
  is_refusal: boolean;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
}

export interface Answer {
  text: string;
  sources: Source[];
  mode: "live" | "offline" | "gate";
  model: string;
  pii_summary: Record<string, number>;
  grounding: Grounding;
  coverage: { sufficient: boolean; reason: string; oov_rate: number; lexical_mass: number };
  tool_calls: ToolCall[];
  usage: Record<string, number>;
  audit_hash: string;
}

export interface Health {
  status: string;
  mode: "live" | "offline";
  model: string | null;
  embedding_provider: string;
  chunks_indexed: number;
  documents: number;
  pii_redaction: boolean;
  audit_chain_valid: boolean;
}

export interface RedactPreview {
  redacted_text: string;
  summary: Record<string, number>;
  findings: { kind: string; placeholder: string; start: number; end: number }[];
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then(json<Health>),
  ask: (question: string, sessionId: string) =>
    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    }).then(json<Answer>),
  redactPreview: (text: string, signal?: AbortSignal) =>
    fetch("/api/redact/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal,
    }).then(json<RedactPreview>),
};

export const PII_LABELS: Record<string, string> = {
  israeli_id: "תעודת זהות",
  company_id: "ח.פ",
  phone: "טלפון",
  email: "כתובת אימייל",
  bank_account: "חשבון בנק",
  credit_card: "כרטיס אשראי",
};
