import type { ToolCall } from "../lib/api";

const TOOL_LABELS: Record<string, string> = {
  calc_vat: "חישוב מע\"מ",
  calc_depreciation: "לוח פחת",
  calc_materiality: "חישוב מהותיות",
  present_value: "ערך נוכחי",
};

/**
 * Every number in the answer came from Python, not from the model. Showing the
 * inputs alongside the result is what makes the answer defensible in a working
 * paper -- the reviewer can see which rate was applied, not just the total.
 */
export function ToolTrace({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;

  return (
    <details className="tool-trace">
      <summary>
        חישובים שבוצעו בקוד ({calls.length})
      </summary>
      {calls.map((c, i) => (
        <div key={i} className="tool-call">
          <div className="tool-name">{TOOL_LABELS[c.name] ?? c.name}</div>
          <table>
            <tbody>
              {Object.entries(c.output).map(([k, v]) => (
                <tr key={k}>
                  <th>{k}</th>
                  <td>{typeof v === "object" ? JSON.stringify(v) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </details>
  );
}
