import { PII_LABELS, type RedactPreview } from "../lib/api";

/**
 * Shows what will be stripped, *before* the user sends.
 *
 * A redactor that works silently gets no credit for working. Users who can watch
 * it catch their client's ID number stop pasting around it, which is the actual
 * goal -- the control only helps if people route their work through it.
 */
export function PiiNotice({ preview }: { preview: RedactPreview | null }) {
  if (!preview || preview.findings.length === 0) return null;

  const parts = Object.entries(preview.summary).map(
    ([kind, n]) => `${n} × ${PII_LABELS[kind] ?? kind}`,
  );

  return (
    <div className="pii-notice" role="status">
      <span className="pii-dot" aria-hidden />
      <span>
        יוסרו לפני השליחה למודל: <strong>{parts.join(" · ")}</strong>
      </span>
      <details>
        <summary>הצג את הטקסט שיישלח</summary>
        <pre>{preview.redacted_text}</pre>
      </details>
    </div>
  );
}
