import type { Source } from "../lib/api";

export function SourcePanel({
  sources,
  highlighted,
}: {
  sources: Source[];
  highlighted: string | null;
}) {
  if (sources.length === 0) return null;

  return (
    <aside className="sources">
      <h2>מקורות</h2>
      <ol>
        {sources.map((s, i) => (
          <li
            key={s.chunk_id}
            id={`src-${s.chunk_id}`}
            className={highlighted === s.chunk_id ? "source highlighted" : "source"}
          >
            <div className="source-head">
              <span className="source-num">{i + 1}</span>
              <span className="source-cite">{s.citation}</span>
              {s.pinned && <span className="badge" title="השאילתה ציינה מזהה מפורש">מזהה מדויק</span>}
            </div>
            <p className="source-text">{s.text}</p>
          </li>
        ))}
      </ol>
    </aside>
  );
}
