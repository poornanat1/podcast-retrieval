"""Evaluate "more like this": list quality without relevance labels.

    uv run python -m ml.evaluation.similar --api http://localhost:8080 [--seeds 300]
        [--json-out experiments/eval/similar.json]

No human judgments exist for episode-to-episode similarity, so this
measures the properties the design asks the endpoint to have, over a
sample of seed episodes, for both index legs:

- intra-list diversity: 1 - mean pairwise cosine similarity among a
  list's episodes (in the same index), so 0 is ten copies and higher is
  more varied;
- novelty: share of results from a show other than the seed's;
- show coverage: distinct shows per list, and distinct shows and
  episodes across all lists over the number recommended;
- repetition: share of results whose normalized title repeats another
  result in the same list (serialized shows, re-uploads);
- popularity concentration: share of results from shows in the top
  decile of popularity, against that decile's share of the catalog;
- similarity to the seed (mean and minimum), and latency.

Seeds are sampled from episodes with at least one judged relevance (the
relevance set's pool), so they are the kind of episode people find.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from ml.datasets.snapshot import DEFAULT_DATABASE_URL

LEGS = ("two_tower", "e5")


def normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def fetch(api: str, episode_id: int, model: str, limit: int) -> dict:
    url = f"{api}/v1/episodes/{episode_id}/similar?model={model}&limit={limit}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def list_metrics(seed: dict, results: list[dict], vectors: dict[int, np.ndarray],
                 top_shows: set[int]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    vecs = [vectors[r["episode_id"]] for r in results if r["episode_id"] in vectors]
    pairs = list(itertools.combinations(vecs, 2))
    ild = 1 - float(np.mean([float(a @ b) for a, b in pairs])) if pairs else None
    shows = [r["podcast_id"] for r in results]
    titles = Counter(normalize_title(r["title"]) for r in results)
    return {
        "n": n,
        "ild": ild,
        "novelty": sum(1 for s in shows if s != seed["podcast_id"]) / n,
        "shows": len(set(shows)),
        "repetition": sum(c - 1 for c in titles.values()) / n,
        "top_decile_share": sum(1 for s in shows if s in top_shows) / n,
        "sim_mean": float(np.mean([r["similarity"] for r in results])),
        "sim_min": float(min(r["similarity"] for r in results)),
    }


E5_MODEL = "intfloat/multilingual-e5-small"


def index_model(conn, leg: str) -> str:
    """The stored model id behind a leg (the two-tower id carries the
    artifact name)."""
    if leg == "e5":
        return E5_MODEL
    with conn.cursor() as cur:
        cur.execute("SELECT model FROM episode_embeddings WHERE model LIKE 'two-tower:%%' LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("no two-tower vectors in the database")
    return row[0]


def summarize(per_list: list[dict], latencies: list[float], failures: int,
              all_eps: set, all_shows: set) -> dict:
    lists = [m for m in per_list if m["n"]]

    def mean(key, rows=lists):
        vals = [m[key] for m in rows if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    lat = sorted(latencies)
    return {
        "lists": len(lists), "empty": len(per_list) - len(lists), "failures": failures,
        "ild": mean("ild"), "novelty": mean("novelty"), "shows_per_list": mean("shows"),
        "repetition": mean("repetition"), "top_decile_share": mean("top_decile_share"),
        "sim_mean": mean("sim_mean"), "sim_min": mean("sim_min"),
        "distinct_episodes": len(all_eps), "distinct_shows": len(all_shows),
        "recommended": sum(m["n"] for m in lists),
        "same_show_shown": mean_of(per_list, "same_show_shown"),
        "dropped": mean_of(per_list, "dropped"),
        "latency_ms": {"p50": pct(lat, .5), "p95": pct(lat, .95)},
    }


def evaluate(api: str, database_url: str, seeds: int, limit: int, rng_seed: int) -> dict:
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.podcast_id FROM episodes e
                WHERE EXISTS (SELECT 1 FROM episode_embeddings m
                              WHERE m.episode_id = e.id AND m.model LIKE 'two-tower:%%')
                  AND e.published_at IS NOT NULL
                ORDER BY md5(e.id::text || %s) LIMIT %s""", (str(rng_seed), seeds))
            seed_rows = [{"episode_id": i, "podcast_id": p} for i, p in cur.fetchall()]
            cur.execute("SELECT id, popularity FROM podcasts")
            pops = cur.fetchall()
        cutoff = sorted((p for _, p in pops), reverse=True)[max(len(pops) // 10 - 1, 0)]
        top_shows = {i for i, p in pops if p >= cutoff and p > 0}
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FILTER (WHERE podcast_id = ANY(%s))::float / count(*) "
                        "FROM episodes", (list(top_shows),))
            top_decile_catalog_share = cur.fetchone()[0]

        out = {"api": api, "seeds": len(seed_rows), "limit": limit,
               "top_decile_catalog_share": top_decile_catalog_share, "legs": {}}
        for leg in LEGS:
            model = index_model(conn, leg)
            per_list, latencies, all_eps, all_shows, failures = [], [], set(), set(), 0
            for seed in seed_rows:
                t0 = time.perf_counter()
                try:
                    resp = fetch(api, seed["episode_id"], leg, limit)
                except Exception as err:  # noqa: BLE001 - count and continue
                    failures += 1
                    print(f"{leg} {seed['episode_id']}: {err}", file=sys.stderr)
                    continue
                latencies.append((time.perf_counter() - t0) * 1000)
                results = resp.get("results") or []
                ids = [r["episode_id"] for r in results]
                vectors, show_of = {}, {}
                if ids:
                    with conn.cursor() as cur:
                        cur.execute("SELECT episode_id, embedding FROM episode_embeddings "
                                    "WHERE episode_id = ANY(%s) AND model = %s", (ids, model))
                        for eid, vec in cur.fetchall():
                            arr = vec.to_numpy() if hasattr(vec, "to_numpy") else vec
                            vectors[eid] = np.asarray(arr, dtype=np.float32)
                        cur.execute("SELECT id, podcast_id FROM episodes WHERE id = ANY(%s)",
                                    (ids,))
                        show_of = dict(cur.fetchall())
                for r in results:
                    r["podcast_id"] = show_of.get(r["episode_id"])
                m = list_metrics(seed, results, vectors, top_shows)
                m["diversification"] = resp.get("diversification")
                per_list.append(m)
                all_eps.update(ids)
                all_shows.update(r["podcast_id"] for r in results)
            out["legs"][leg] = summarize(per_list, latencies, failures, all_eps, all_shows)
        return out


def mean_of(per_list, key):
    vals = [m["diversification"][key] for m in per_list if m.get("diversification")]
    return float(np.mean(vals)) if vals else None


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--database-url",
                        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    report = evaluate(args.api, args.database_url, args.seeds, args.limit, args.seed)
    for leg, m in report["legs"].items():
        print(f"{leg:10s} lists={m['lists']} ild={m['ild']:.3f} novelty={m['novelty']:.2f} "
              f"shows/list={m['shows_per_list']:.1f} repetition={m['repetition']:.3f} "
              f"top-decile={m['top_decile_share']:.2f} "
              f"(catalog {report['top_decile_catalog_share']:.2f}) "
              f"sim={m['sim_mean']:.3f}/{m['sim_min']:.3f} p50={m['latency_ms']['p50']:.0f}ms "
              f"p95={m['latency_ms']['p95']:.0f}ms")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
