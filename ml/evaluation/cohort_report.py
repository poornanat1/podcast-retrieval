"""Per-cohort comparison tables with regressions against baselines flagged.

    uv run python -m ml.evaluation.cohort_report --summaries compare.json \
        --baselines vector-e5-small,lexical-fts [--labels name=label,...] \
        [--tolerance 0.05] [--out report.md]

Reads the summaries ``ml.evaluation.compare --json-out`` writes (one judgment
pool, one run). It prints three tables:

- NDCG@10 by query type;
- NDCG@10 by query language;
- Recall@100 by relevant-item cohort: transcript availability, head/tail
  show, and, when the comparison ran with ``--cold-start-dataset``, the
  cold-start cohorts (training positive, new episode of a known show, show
  unseen in training, added after the training snapshot).

Each table has a query count per cohort, because a cohort of 20 queries
moves a lot on one query. A cell is marked ``▼`` when the system is more
than ``--tolerance`` below any baseline, on cohorts with at least
``--min-queries`` queries; smaller cohorts are shown but never flagged. The
regressions are listed under the tables with the baseline they were
measured against. Baselines are never flagged against each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ITEM_COHORTS = ("transcript", "no_transcript", "head", "tail",
                "seen", "known_podcast", "unseen_podcast", "post_snapshot")
COHORT_K = 100


def cohort_values(summary: dict) -> dict[str, dict[str, tuple[float, int]]]:
    """``{section: {cohort: (value, queries)}}`` for one system's summary."""
    out: dict[str, dict[str, tuple[float, int]]] = {"query type": {}, "query language": {},
                                                    "relevant items": {}}
    for name, values in summary.get("by_query_type", {}).items():
        out["query type"][name] = (values["ndcg_at_10"], int(values.get("queries", 0)))
    for name, values in summary.get("by_query_language", {}).items():
        out["query language"][name] = (values["ndcg_at_10"], int(values.get("queries", 0)))
    g = summary.get("global", {})
    for name in ITEM_COHORTS:
        key = f"recall_at_{COHORT_K}__{name}"
        if key in g:
            out["relevant items"][name] = (g[key], int(g.get(f"{key}__queries", 0)))
    return out


def regressions(values: dict, baselines: dict[str, dict], tolerance: float,
                min_queries: int = 10) -> list[dict]:
    """Every (section, cohort, baseline) where the system trails by more than
    ``tolerance`` on a cohort of at least ``min_queries`` queries (an unknown
    count, 0, is not filtered)."""
    found = []
    for section, cohorts in values.items():
        for cohort, (value, queries) in cohorts.items():
            if 0 < queries < min_queries:
                continue
            for base_name, base_values in baselines.items():
                base = base_values.get(section, {}).get(cohort)
                if base is not None and value < base[0] - tolerance:
                    found.append({"section": section, "cohort": cohort, "baseline": base_name,
                                  "value": value, "baseline_value": base[0],
                                  "delta": round(value - base[0], 4)})
    return found


def render(summaries: list[dict], baselines: list[str], labels: dict[str, str],
           tolerance: float = 0.05, min_queries: int = 10) -> str:
    by_name = {s["system"]: s for s in summaries}
    missing = [b for b in baselines if b not in by_name]
    if missing:
        raise ValueError(f"baselines not in summaries: {missing}")
    order = [s["system"] for s in summaries]
    values = {name: cohort_values(by_name[name]) for name in order}
    base_values = {b: values[b] for b in baselines}
    flagged = {name: regressions(values[name], base_values, tolerance, min_queries)
               for name in order if name not in baselines}

    def label(name: str) -> str:
        return labels.get(name, name)

    metric = {"query type": "NDCG@10", "query language": "NDCG@10",
              "relevant items": f"Recall@{COHORT_K}"}
    lines: list[str] = []
    for section in ("query type", "query language", "relevant items"):
        cohorts = list(dict.fromkeys(c for name in order for c in values[name][section]))
        if not cohorts:
            continue
        counts = {c: max(values[name][section].get(c, (0, 0))[1] for name in order)
                  for c in cohorts}
        lines.append(f"{metric[section]} by {section}:\n")
        lines.append("| system | " + " | ".join(
            f"{c} ({counts[c]})" if counts[c] else c for c in cohorts) + " |")
        lines.append("|---|" + "---|" * len(cohorts))
        for name in order:
            marks = {(r["section"], r["cohort"]) for r in flagged.get(name, [])}
            cells = []
            for c in cohorts:
                v = values[name][section].get(c)
                cell = "—" if v is None else f"{v[0]:.3f}"
                if (section, c) in marks:
                    cell += " ▼"
                cells.append(cell)
            lines.append(f"| {label(name)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(f"Regressions beyond {tolerance} against "
                 + ", ".join(label(b) for b in baselines)
                 + f" (cohorts of at least {min_queries} queries):\n")
    any_found = False
    for name, found in flagged.items():
        for r in found:
            any_found = True
            lines.append(f"- {label(name)} vs {label(r['baseline'])}: {r['cohort']} "
                         f"({r['section']}) {r['value']:.3f} vs {r['baseline_value']:.3f} "
                         f"({r['delta']:+.3f})")
    if not any_found:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--baselines", default="vector-e5-small,lexical-fts")
    parser.add_argument("--labels", default="", help="system=label pairs, comma-separated")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--min-queries", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    summaries = json.loads(Path(args.summaries).read_text())
    labels = dict(pair.split("=", 1) for pair in args.labels.split(",") if pair)
    text = render(summaries, [b for b in args.baselines.split(",") if b], labels,
                  args.tolerance, args.min_queries)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
