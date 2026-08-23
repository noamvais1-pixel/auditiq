"""Evaluation harness. Run with `make eval`.

The point of this file is to make claims about the system falsifiable. "The RAG
works well" is not a claim; "recall@6 is 0.87 on paraphrased questions and 0.62
with the offline embedding provider" is.

Two suites, because they fail independently:

*Retrieval.* Recall@k and MRR against hand-labelled gold chunks. Questions are
tagged ``retrieval`` (the answer's keywords appear in the question),
``paraphrase`` (they do not -- this is where lexical-only retrieval falls over),
and ``multi`` (the answer spans several clauses). Reporting one blended number
across all three hides exactly the weakness you need to see.

*Grounding.* Does the system refuse when the corpus does not cover the question,
and are its citations real? ``out_of_scope`` questions have no gold chunk: the
only correct behaviour is a refusal. A system that scores well on retrieval and
answers all three of these anyway is worse than useless in an audit file, so
this suite gates separately.

Exit code is non-zero if any suite falls below its threshold, so this runs in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.pipeline import Assistant  # noqa: E402
from app.rag import Index, load_corpus  # noqa: E402

DATASET = Path(__file__).with_name("dataset.jsonl")

# The refusal metric is split by which layer of the coverage gate is expected to
# catch each case (see app/rag/coverage.py). The lexical layer runs everywhere and
# is gated unconditionally. The model layer only exists when an API key is present,
# so offline runs report it as skipped rather than quietly passing -- a green CI
# run that never exercised the semantic gate would be worse than a red one.
THRESHOLDS = {
    "recall@6": 0.75,
    "mrr": 0.55,
    "refusal_rate_lexical": 1.0,
    "refusal_rate_model": 1.0,   # enforced only in live mode
    "citation_validity": 1.0,    # no fabricated ids, ever
}


def load_cases() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_retrieval(index: Index, cases: list[dict], k: int) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    rows = []

    for case in cases:
        if case["kind"] == "out_of_scope":
            continue
        hits = index.search(case["question"], k=k)
        got = [h.chunk.chunk_id for h in hits]
        gold = set(case["gold"])

        found = gold & set(got)
        recall = len(found) / len(gold)
        rank = next((i + 1 for i, cid in enumerate(got) if cid in gold), None)

        row = {
            "id": case["id"],
            "kind": case["kind"],
            "recall": recall,
            "rr": 1 / rank if rank else 0.0,
            "first_rank": rank,
            "missed": sorted(gold - found),
            "top": got[:3],
        }
        rows.append(row)
        by_kind[case["kind"]].append(row)

    def agg(rs: list[dict]) -> dict:
        n = len(rs) or 1
        return {
            "n": len(rs),
            f"recall@{k}": round(sum(r["recall"] for r in rs) / n, 3),
            "mrr": round(sum(r["rr"] for r in rs) / n, 3),
            "misses": [r["id"] for r in rs if r["recall"] < 1.0],
        }

    return {
        "overall": agg(rows),
        "by_kind": {kind: agg(rs) for kind, rs in sorted(by_kind.items())},
        "rows": rows,
    }


def eval_grounding(assistant: Assistant, cases: list[dict]) -> dict:
    oos = [c for c in cases if c["kind"] == "out_of_scope"]
    in_scope = [c for c in cases if c["kind"] != "out_of_scope"]
    live = assistant.llm.mode == "live"

    by_gate: dict[str, list[dict]] = defaultdict(list)
    degraded: list[str] = []
    for case in oos:
        ans = assistant.ask(case["question"], session_id="eval")
        if ans.mode == "degraded":
            degraded.append(case["id"])
        refused = bool(ans.grounding["blocked"] or ans.grounding["is_refusal"] or not ans.sources)
        by_gate[case.get("gate", "lexical")].append(
            {"id": case["id"], "refused": refused, "stopped_at": ans.mode}
        )

    def rate(rows: list[dict]) -> float | None:
        return round(sum(r["refused"] for r in rows) / len(rows), 3) if rows else None

    fabricated, checked = 0, 0
    for case in in_scope[:10]:  # sampled -- this half costs tokens in live mode
        ans = assistant.ask(case["question"], session_id="eval")
        if ans.mode == "degraded":
            degraded.append(case["id"])
        fabricated += len(ans.grounding["invalid_citations"])
        checked += len(ans.grounding["valid_citations"]) + len(ans.grounding["invalid_citations"])

    return {
        "live": live,
        # A live run where some answers fell back to the extractive path did not
        # measure what it claims to have measured. Surfaced, and gated, rather
        # than averaged into the score.
        "degraded_answers": degraded,
        "refusal_rate_lexical": rate(by_gate["lexical"]),
        "refusal_rate_model": rate(by_gate["model"]) if live else None,
        "model_gate_skipped": not live,
        "refusal_detail": {g: rows for g, rows in by_gate.items()},
        "citations_checked": checked,
        "fabricated_citations": fabricated,
        "citation_validity": round((checked - fabricated) / checked, 3) if checked else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=settings.top_k)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-gate", action="store_true", help="report only, always exit 0")
    args = parser.parse_args()

    cases = load_cases()
    index = Index().build(load_corpus(settings.corpus_dir))
    assistant = Assistant()

    retrieval = eval_retrieval(index, cases, args.k)
    grounding = eval_grounding(assistant, cases)

    report = {
        "embedding_provider": index.provider.name,
        # Recorded because a score is only reproducible if you know what produced
        # it -- and because free-tier daily caps mean the model used for a given
        # run is not always the configured default.
        "llm_provider": assistant.llm.provider_name,
        "llm_model": assistant.llm.model,
        "llm_mode": assistant.llm.mode,
        "k": args.k,
        "retrieval": retrieval,
        "grounding": grounding,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report, args.k)

    failures = _check(report, args.k)
    for line in failures:
        print(f"FAIL {line}")
    return 0 if (args.no_gate or not failures) else 1


def _print_report(report: dict, k: int) -> None:
    r = report["retrieval"]
    g = report["grounding"]
    print(f"\nembeddings={report['embedding_provider']}  "
          f"llm={report['llm_provider']}/{report['llm_model']} ({report['llm_mode']})  k={k}")
    print("\n-- retrieval " + "-" * 46)
    print(f"{'kind':<14}{'n':>4}{'recall@'+str(k):>12}{'mrr':>8}")
    for kind, stats in r["by_kind"].items():
        print(f"{kind:<14}{stats['n']:>4}{stats[f'recall@{k}']:>12}{stats['mrr']:>8}")
    o = r["overall"]
    print(f"{'ALL':<14}{o['n']:>4}{o[f'recall@{k}']:>12}{o['mrr']:>8}")

    if o["misses"]:
        print("\nmisses:")
        for row in r["rows"]:
            if row["recall"] < 1.0:
                print(f"  {row['id']:<22} missed={','.join(row['missed'])}")
                print(f"  {'':<22} top3={row['top']}")

    print("\n-- grounding " + "-" * 46)
    print(f"refusal, lexical gate   : {g['refusal_rate_lexical']}")
    model_rate = "SKIPPED (offline — no API key)" if g["model_gate_skipped"] else g["refusal_rate_model"]
    print(f"refusal, model gate     : {model_rate}")
    for gate, rows in g["refusal_detail"].items():
        for row in rows:
            if not row["refused"]:
                print(f"    [{gate}] answered instead of refusing: {row['id']}")
    print(f"citation validity       : {g['citation_validity']} "
          f"({g['fabricated_citations']} fabricated / {g['citations_checked']} checked)")
    if g["degraded_answers"]:
        print(f"degraded to fallback    : {len(g['degraded_answers'])} "
              f"-> {', '.join(g['degraded_answers'])}")
    print()


def _check(report: dict, k: int) -> list[str]:
    out = []
    o = report["retrieval"]["overall"]
    if o[f"recall@{k}"] < THRESHOLDS["recall@6"]:
        out.append(f"recall@{k}={o[f'recall@{k}']} < {THRESHOLDS['recall@6']}")
    if o["mrr"] < THRESHOLDS["mrr"]:
        out.append(f"mrr={o['mrr']} < {THRESHOLDS['mrr']}")
    g = report["grounding"]
    lex = g["refusal_rate_lexical"]
    if lex is not None and lex < THRESHOLDS["refusal_rate_lexical"]:
        out.append(f"refusal_rate_lexical={lex} < {THRESHOLDS['refusal_rate_lexical']}")
    mod = g["refusal_rate_model"]
    if mod is not None and mod < THRESHOLDS["refusal_rate_model"]:
        out.append(f"refusal_rate_model={mod} < {THRESHOLDS['refusal_rate_model']}")
    if g["degraded_answers"]:
        out.append(
            f"{len(g['degraded_answers'])} answers fell back to the extractive path; "
            "this run did not measure the model"
        )
    if g["citation_validity"] < THRESHOLDS["citation_validity"]:
        out.append(f"citation_validity={g['citation_validity']} < {THRESHOLDS['citation_validity']}")
    return out


if __name__ == "__main__":
    sys.exit(main())
