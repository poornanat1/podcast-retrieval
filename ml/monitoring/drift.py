"""Drift check: compare recent serving behaviour with a recorded baseline.

    uv run python -m ml.monitoring.drift [--hours 6] [--since ISO-TIME]
        [--baseline experiments/monitoring/baseline.json] [--prometheus http://localhost:9090]
        [--webhook URL] [--json-out FILE] [--write-baseline]

Reads the stored searches of the last ``--hours`` (the ``searches`` table:
legs, timings, ordering, result scores and sources) and, when a Prometheus
URL is given, the model server's per-feature sums and counts over the same
window. It compares them with the baseline and reports findings:

- zero-result rate, retrieval leg failures, reranker fallback (RRF)
- latency p95 per stage
- the distribution of top scores and of result sources (population
  stability index, PSI)
- feature means at serving against the baseline means, and NaN counts
- the served bundle

Exit code 2 and a webhook post (Slack-compatible ``{"text": ...}``) when
anything drifted (0 with ``--exit-zero``, which Railway's cron needs: a
non-zero exit is a crash there and a crashed cron is not rescheduled);
0 otherwise; 1 on error. ``--write-baseline`` records the
window as the new baseline instead. Runs as a cron job on Railway
(deploy: the ``drift`` service) and locally against the compose stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("experiments/monitoring/baseline.json")
LEGS = ("lexical", "e5", "two_tower")
SCORE_BINS = [-4 + 0.5 * i for i in range(29)]  # -4 .. 10
STAGES = ("total_ms", "encode_ms", "retrieve_ms", "evidence_ms", "rerank_ms")

THRESHOLDS = {
    "min_searches": 100,  # a handful of real searches says nothing about distributions
    "zero_result_rate_delta": 0.15,  # absolute increase over the baseline
    "vector_leg_failure_rate": 0.10,  # e5 / two_tower not ok
    "lexical_timeout_rate_delta": 0.30,  # lexical times out routinely; watch the change
    "rrf_rate": 0.05,
    "p95_ratio": 2.0,  # and above p95_floor_ms
    "p95_floor_ms": 1000.0,
    "psi": 0.25,
    "feature_shift": 0.5,  # relative change of a feature mean
    "feature_min_count": 200,
}


@dataclass
class Finding:
    name: str
    severity: str  # critical | warning | info
    message: str
    value: float | None = None
    limit: float | None = None


# --- statistics -------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def histogram(values: list[float], bins: list[float]) -> list[float]:
    """Fractions per bin; the last bin collects everything at or above the
    last edge, the first everything below the second edge."""
    counts = [0] * len(bins)
    for v in values:
        i = 0
        while i + 1 < len(bins) and v >= bins[i + 1]:
            i += 1
        counts[i] += 1
    n = max(len(values), 1)
    return [c / n for c in counts]


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    """Population stability index between two fraction vectors."""
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


# --- summarising stored searches ----------------------------------------------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Serving statistics from ``searches`` rows (legs, timings_ms, results,
    ordering, bundle_id, input)."""
    n = len(rows)
    zero = sum(1 for r in rows if not r.get("results"))
    rrf = sum(1 for r in rows if r.get("ordering") != "ranker")
    leg_status: dict[str, dict[str, int]] = {leg: {} for leg in LEGS}
    timings: dict[str, list[float]] = {s: [] for s in STAGES}
    top_scores: list[float] = []
    source_counts: dict[str, int] = {"lexical": 0, "e5": 0, "two_tower": 0}
    bundles: dict[str, int] = {}
    inputs: dict[str, int] = {}
    for r in rows:
        for leg, res in (r.get("legs") or {}).items():
            if leg in leg_status and res:
                status = res.get("status", "unknown")
                leg_status[leg][status] = leg_status[leg].get(status, 0) + 1
        for stage, ms in (r.get("timings_ms") or {}).items():
            if stage in timings and ms is not None:
                timings[stage].append(float(ms))
        results = r.get("results") or []
        if results:
            top_scores.append(float(results[0].get("score", 0.0)))
        for item in results:
            for s in item.get("sources") or []:
                source_counts[s] = source_counts.get(s, 0) + 1
        b = r.get("bundle_id") or "unknown"
        bundles[b] = bundles.get(b, 0) + 1
        mode = ((r.get("input") or {}).get("mode")) or "text"
        inputs[mode] = inputs.get(mode, 0) + 1
    total_sources = max(sum(source_counts.values()), 1)
    return {
        "searches": n,
        "zero_result_rate": zero / n if n else None,
        "rrf_rate": rrf / n if n else None,
        "leg_status": {leg: {s: c / n for s, c in st.items()} for leg, st in leg_status.items()}
        if n else {},
        "timings_ms": {s: {"p50": percentile(v, 0.5), "p95": percentile(v, 0.95), "n": len(v)}
                       for s, v in timings.items()},
        "top_score_hist": histogram(top_scores, SCORE_BINS),
        "top_score_mean": sum(top_scores) / len(top_scores) if top_scores else None,
        "source_mix": {s: c / total_sources for s, c in source_counts.items()},
        "bundles": bundles,
        "inputs": inputs,
    }


def leg_rate(summary: dict[str, Any], leg: str, statuses: tuple[str, ...]) -> float:
    st = (summary.get("leg_status") or {}).get(leg) or {}
    return sum(st.get(s, 0.0) for s in statuses)


# --- comparison ---------------------------------------------------------------

def compare(baseline: dict[str, Any], current: dict[str, Any],
            thresholds: dict[str, float] | None = None) -> list[Finding]:
    t = {**THRESHOLDS, **(thresholds or {})}
    out: list[Finding] = []
    n = current.get("searches", 0)
    if n < t["min_searches"]:
        out.append(Finding("sample", "info", f"only {n} searches in the window; "
                           f"needs {int(t['min_searches'])} to judge drift"))
        return out

    base_zero = baseline.get("zero_result_rate") or 0.0
    if current["zero_result_rate"] > base_zero + t["zero_result_rate_delta"]:
        out.append(Finding("zero_results", "warning",
                           f"zero-result rate {current['zero_result_rate']:.0%} "
                           f"(baseline {base_zero:.0%})",
                           current["zero_result_rate"], base_zero + t["zero_result_rate_delta"]))

    for leg in ("e5", "two_tower"):
        rate = leg_rate(current, leg, ("timeout", "error"))
        if rate > t["vector_leg_failure_rate"]:
            out.append(Finding(f"leg_{leg}", "critical",
                               f"{leg} leg failed or timed out on {rate:.0%} of searches",
                               rate, t["vector_leg_failure_rate"]))
    lex = leg_rate(current, "lexical", ("timeout", "error"))
    base_lex = leg_rate(baseline, "lexical", ("timeout", "error"))
    if lex > base_lex + t["lexical_timeout_rate_delta"]:
        out.append(Finding("leg_lexical", "warning",
                           f"lexical leg timed out on {lex:.0%} (baseline {base_lex:.0%})",
                           lex, base_lex + t["lexical_timeout_rate_delta"]))

    if current["rrf_rate"] > t["rrf_rate"]:
        out.append(Finding("rrf_fallback", "critical",
                           f"{current['rrf_rate']:.0%} of searches fell back to RRF ordering",
                           current["rrf_rate"], t["rrf_rate"]))

    for stage in STAGES:
        cur = (current["timings_ms"].get(stage) or {}).get("p95")
        base = ((baseline.get("timings_ms") or {}).get(stage) or {}).get("p95")
        if cur is None or base is None:
            continue
        if cur > t["p95_floor_ms"] and cur > base * t["p95_ratio"]:
            out.append(Finding(f"latency_{stage}", "warning",
                               f"{stage} p95 {cur:.0f} ms (baseline {base:.0f} ms)",
                               cur, base * t["p95_ratio"]))

    if baseline.get("top_score_hist") and current.get("top_score_hist"):
        v = psi(baseline["top_score_hist"], current["top_score_hist"])
        if v > t["psi"]:
            out.append(Finding("top_score_psi", "critical",
                               f"top-score distribution PSI {v:.2f} "
                               f"(mean {current['top_score_mean']:.2f} vs "
                               f"{baseline.get('top_score_mean', float('nan')):.2f})",
                               v, t["psi"]))
    if baseline.get("source_mix") and current.get("source_mix"):
        keys = sorted(set(baseline["source_mix"]) | set(current["source_mix"]))
        v = psi([baseline["source_mix"].get(k, 0.0) for k in keys],
                [current["source_mix"].get(k, 0.0) for k in keys])
        if v > t["psi"]:
            mix = ", ".join(f"{k} {current['source_mix'].get(k, 0):.0%}" for k in keys)
            out.append(Finding("source_mix_psi", "critical",
                               f"result-source mix PSI {v:.2f} ({mix})", v, t["psi"]))

    base_bundles = set(baseline.get("bundles") or {})
    new_bundles = set(current.get("bundles") or {}) - base_bundles
    if base_bundles and new_bundles:
        out.append(Finding("bundle", "info",
                           f"served bundle changed: {', '.join(sorted(new_bundles))} "
                           f"(baseline {', '.join(sorted(base_bundles))})"))

    base_feats = baseline.get("features") or {}
    cur_feats = current.get("features") or {}
    for name, cur in cur_feats.items():
        base = base_feats.get(name)
        if not base or cur.get("count", 0) < t["feature_min_count"]:
            continue
        if cur.get("nan", 0) > 0:
            out.append(Finding(f"feature_nan_{name}", "critical",
                               f"{name}: {int(cur['nan'])} non-finite values", cur["nan"], 0))
        bm, cm = base.get("mean"), cur.get("mean")
        if bm is None or cm is None:
            continue
        scale = max(abs(bm), 1e-3)
        shift = abs(cm - bm) / scale
        if shift > t["feature_shift"]:
            out.append(Finding(f"feature_{name}", "warning",
                               f"{name} mean {cm:.3g} vs baseline {bm:.3g}", shift,
                               t["feature_shift"]))
    return out


# --- data sources ---------------------------------------------------------------

def load_rows(database_url: str, hours: float,
              since: datetime | None = None) -> list[dict[str, Any]]:
    """Searches of the last ``hours``, or since a fixed time (for replays
    and demonstrations)."""
    import psycopg

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        if since is not None:
            cur.execute("""SELECT ordering, bundle_id, legs, timings_ms, results, input
                           FROM searches WHERE created_at >= %s""", (since,))
        else:
            cur.execute(
                """SELECT ordering, bundle_id, legs, timings_ms, results, input
                   FROM searches WHERE created_at >= now() - make_interval(secs => %s)""",
                (hours * 3600,))
        return [{"ordering": o, "bundle_id": b, "legs": legs, "timings_ms": tm,
                 "results": res, "input": inp} for o, b, legs, tm, res, inp in cur.fetchall()]


def prometheus_query(url: str, expr: str) -> dict[str, float]:
    q = urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(f"{url.rstrip('/')}/api/v1/query?{q}", timeout=30) as resp:
        body = json.load(resp)
    out = {}
    for item in body.get("data", {}).get("result", []):
        key = item["metric"].get("feature", "")
        value = float(item["value"][1])
        if math.isfinite(value):
            out[key] = value
    return out


def load_features(url: str, hours: float) -> dict[str, dict[str, float]]:
    window = f"{int(hours * 60)}m"

    def over(metric: str) -> dict[str, float]:
        return prometheus_query(url, f"sum by (feature) (increase({metric}[{window}]))")

    pos = over("podfind_model_feature_positive_sum_total")
    neg = over("podfind_model_feature_negative_sum_total")
    counts = over("podfind_model_feature_values_total")
    nans = over("podfind_model_feature_nan_total")
    out = {}
    for name, count in counts.items():
        if count <= 0:
            continue
        out[name] = {"mean": (pos.get(name, 0.0) - neg.get(name, 0.0)) / count, "count": count,
                     "nan": nans.get(name, 0.0)}
    return out


def notify(webhook: str, text: str) -> None:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30):
        pass


def format_report(findings: list[Finding], current: dict[str, Any], hours: float) -> str:
    head = (f"PodFind drift check: {len([f for f in findings if f.severity != 'info'])} "
            f"finding(s) over the last {hours:g} h ({current.get('searches', 0)} searches)")
    lines = [head] + [f"- [{f.severity}] {f.message}" for f in findings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", default=os.environ.get(
        "DATABASE_URL", "postgres://podfind:podfind@localhost:5432/podfind"))
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--since", help="ISO time; window start instead of --hours")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--prometheus", default=os.environ.get("PROMETHEUS_URL"))
    parser.add_argument("--webhook", default=os.environ.get("ALERT_WEBHOOK_URL"))
    parser.add_argument("--json-out")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--exit-zero", action="store_true",
                        help="report drift in the output and webhook but exit 0 (for schedulers "
                             "that treat a non-zero exit as a crash and stop rescheduling)")
    args = parser.parse_args(argv)

    since = datetime.fromisoformat(args.since) if args.since else None
    if since is not None:
        args.hours = max((datetime.now(UTC) - since).total_seconds() / 3600, 1 / 60)
    rows = load_rows(args.database_url, args.hours, since)
    current = summarize(rows)
    current["window_hours"] = args.hours
    current["computed_at"] = datetime.now(UTC).isoformat()
    if args.prometheus:
        try:
            current["features"] = load_features(args.prometheus, args.hours)
        except Exception as err:  # noqa: BLE001 - the check must still run
            current["features_error"] = str(err)
            print(f"prometheus unavailable: {err}", file=sys.stderr)

    if args.write_baseline:
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.baseline).write_text(json.dumps(current, indent=2) + "\n")
        print(f"baseline written to {args.baseline} from {current['searches']} searches")
        return 0

    baseline = json.loads(Path(args.baseline).read_text())
    findings = compare(baseline, current)
    report = format_report(findings, current, args.hours)
    print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"findings": [asdict(f) for f in findings], "current": current,
             "baseline_computed_at": baseline.get("computed_at")}, indent=2) + "\n")
    drifted = any(f.severity in ("critical", "warning") for f in findings)
    if drifted and args.webhook and not args.webhook.endswith("/unconfigured"):
        notify(args.webhook, report)
    if drifted and args.exit_zero:
        print("DRIFT (exit 0 requested)")
        return 0
    return 2 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
