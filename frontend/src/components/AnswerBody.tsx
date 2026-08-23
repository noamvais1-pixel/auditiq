import type { Source } from "../lib/api";

const CITATION = /\[\[([^\[\]]+?)\]\]/g;

/**
 * Renders [[chunk_id]] markers as numbered, clickable citation chips.
 *
 * Showing the raw id would be noise to an accountant, but dropping it entirely
 * would break the one property the whole system is built around -- that every
 * claim can be traced to a clause. So each marker becomes a superscript number
 * that scrolls its source into view.
 */
export function AnswerBody({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: Source[];
  onCite: (chunkId: string) => void;
}) {
  const order = new Map(sources.map((s, i) => [s.chunk_id, i + 1]));

  return (
    <div className="answer-body">
      {text.split("\n\n").map((para, pi) => (
        <p key={pi}>{renderParagraph(para, order, onCite)}</p>
      ))}
    </div>
  );
}

function renderParagraph(
  para: string,
  order: Map<string, number>,
  onCite: (id: string) => void,
) {
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  CITATION.lastIndex = 0;

  while ((match = CITATION.exec(para)) !== null) {
    const [full, rawId] = match;
    const id = rawId!.trim();
    if (match.index > cursor) nodes.push(para.slice(cursor, match.index));

    const n = order.get(id);
    nodes.push(
      <button
        key={`${id}-${match.index}`}
        className="citation-chip"
        onClick={() => onCite(id)}
        title={id}
        aria-label={`מקור ${n ?? id}`}
      >
        {n ?? "•"}
      </button>,
    );
    cursor = match.index + full.length;
  }
  if (cursor < para.length) nodes.push(para.slice(cursor));
  return nodes;
}
