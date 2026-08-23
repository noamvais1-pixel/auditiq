import { useCallback, useEffect, useRef, useState } from "react";
import { AnswerBody } from "./components/AnswerBody";
import { PiiNotice } from "./components/PiiNotice";
import { SourcePanel } from "./components/SourcePanel";
import { ToolTrace } from "./components/ToolTrace";
import { api, type Answer, type Health, type RedactPreview } from "./lib/api";

const EXAMPLES = [
  "מאיזה רגע מתחילים לרשום פחת על נכס?",
  "החברה כמעט באיזון תפעולי. איזה מדד מהותיות לבחור?",
  "ספק שמהווה 60% מהרכש שלנו — הוא צד קשור?",
  "שכרתי משרד לתשעה חודשים. חייב להכיר בנכס זכות שימוש?",
];

const SESSION_ID = crypto.randomUUID();

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [preview, setPreview] = useState<RedactPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError("השרת אינו זמין"));
  }, []);

  // Debounced, and the previous request is aborted -- otherwise a fast typist
  // gets responses landing out of order and the notice flickers.
  useEffect(() => {
    if (question.trim().length < 8) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    const timer = setTimeout(() => {
      api
        .redactPreview(question, controller.signal)
        .then(setPreview)
        .catch(() => {});
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [question]);

  const submit = useCallback(async () => {
    const q = question.trim();
    if (q.length < 2 || busy) return;
    setBusy(true);
    setError(null);
    try {
      setAnswer(await api.ask(q, SESSION_ID));
    } catch (e) {
      setError(e instanceof Error ? e.message : "שגיאה לא צפויה");
    } finally {
      setBusy(false);
    }
  }, [question, busy]);

  const onCite = useCallback((chunkId: string) => {
    setHighlighted(chunkId);
    document.getElementById(`src-${chunkId}`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, []);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>AuditIQ</h1>
          <p>עוזר מבוסס מקורות לרואי חשבון ומבקרים</p>
        </div>
        {health && <StatusStrip health={health} />}
      </header>

      {health?.mode === "offline" && (
        <div className="banner" role="status">
          <strong>מצב הדגמה.</strong> לא הוגדר מפתח API, ולכן התשובות מורכבות מקטעי
          המקור שאותרו ללא ניסוח מחדש. האחזור, הסרת המידע המזהה, אכיפת הציטוט ולוג
          הביקורת פועלים במלואם.
        </div>
      )}

      <main>
        <section className="ask">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            placeholder="שאל/י שאלה מקצועית. אפשר להדביק גם קטע מהתיק — נתונים מזהים יוסרו לפני השליחה."
            rows={4}
            aria-label="שאלה"
          />
          <PiiNotice preview={preview} />

          <div className="controls">
            <button onClick={submit} disabled={busy || question.trim().length < 2}>
              {busy ? "מחפש במקורות…" : "שאל"}
            </button>
            <span className="hint">⌘/Ctrl + Enter</span>
          </div>

          {!answer && (
            <div className="examples">
              <span>לדוגמה:</span>
              {EXAMPLES.map((ex) => (
                <button key={ex} className="example" onClick={() => setQuestion(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          )}

          {error && <div className="error" role="alert">{error}</div>}
        </section>

        {answer && (
          <section className="result">
            {answer.mode === "gate" && (
              <div className="refusal-tag">
                נעצר בבדיקת כיסוי — {answer.coverage.reason}
              </div>
            )}
            <AnswerBody text={answer.text} sources={answer.sources} onCite={onCite} />
            <ToolTrace calls={answer.tool_calls} />
            <AnswerFooter answer={answer} />
          </section>
        )}

        {answer && <SourcePanel sources={answer.sources} highlighted={highlighted} />}
      </main>
    </div>
  );
}

function StatusStrip({ health }: { health: Health }) {
  return (
    <dl className="status">
      <div>
        <dt>מצב</dt>
        <dd>{health.mode === "live" ? health.model : "הדגמה"}</dd>
      </div>
      <div>
        <dt>מאגר</dt>
        <dd>
          {health.documents} מסמכים · {health.chunks_indexed} קטעים
        </dd>
      </div>
      <div>
        <dt>אחזור</dt>
        <dd>{health.embedding_provider}</dd>
      </div>
      <div>
        <dt>לוג ביקורת</dt>
        <dd className={health.audit_chain_valid ? "ok" : "bad"}>
          {health.audit_chain_valid ? "שרשרת תקינה" : "שרשרת נשברה"}
        </dd>
      </div>
    </dl>
  );
}

function AnswerFooter({ answer }: { answer: Answer }) {
  const pii = Object.values(answer.pii_summary).reduce((a, b) => a + b, 0);
  return (
    <footer className="answer-footer">
      {pii > 0 && <span>{pii} פרטים מזהים הוסרו לפני השליחה</span>}
      {answer.grounding.uncited_claims.length > 0 && (
        <span className="warn">
          {answer.grounding.uncited_claims.length} טענות ללא ציטוט — יש לאמת מול המקור
        </span>
      )}
      {answer.grounding.invalid_citations.length > 0 && (
        <span className="warn">
          {answer.grounding.invalid_citations.length} ציטוטים שאינם קיימים הוסרו
        </span>
      )}
      <span className="mono" title="מזהה הרשומה בלוג הביקורת">
        {answer.audit_hash.slice(0, 12)}
      </span>
    </footer>
  );
}
