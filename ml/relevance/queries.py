"""Build the relevance-set query file.

The set mixes four query kinds over the seeded catalog:

- navigational and entity queries generated deterministically from the
  catalog snapshot (top shows and person-like publishers), because those
  queries only make sense against shows that exist in the catalog;
- curated exploratory queries (realistic, open-ended, voice-style included);
- curated filtered queries carrying structured constraints (language,
  duration, recency, explicit content) that retrieval must enforce as hard
  predicates.

    uv run python -m ml.relevance.queries --snapshot data/snapshots \
        --out data/relevance/queries.jsonl

Output is deterministic given the snapshot: stable ids, stable order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from ml.datasets import snapshot as snapshot_mod
from ml.relevance.contracts import QueryModel, QueryType

_WORD_RE = re.compile(r"[^0-9a-z]+")
_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}$")

# Capitalized-words publishers that are organizations, not people.
_ORG_WORDS = {
    "media", "network", "networks", "studios", "studio", "productions",
    "production", "radio", "news", "podcasts", "podcast", "entertainment",
    "company", "public", "press", "audio", "broadcasting", "inc", "llc",
}


def _looks_like_person(publisher: str) -> bool:
    if not _PERSON_RE.match(publisher):
        return False
    return not any(word in _ORG_WORDS for word in publisher.lower().split())

NAVIGATIONAL_COUNT = 40
ENTITY_COUNT = 20

# Realistic open-ended queries, spanning the seeded topic mix. Kept short to
# long, including voice-style phrasings.
EXPLORATORY = [
    "history of the roman empire",
    "how did the french revolution start",
    "world war two pacific theater",
    "ancient egypt pharaohs and pyramids",
    "the fall of the berlin wall",
    "unsolved disappearances national parks",
    "cold cases finally solved with dna",
    "con artists and financial fraud stories",
    "cult documentaries and survivor stories",
    "how detectives actually solve murders",
    "practical uses of artificial intelligence",
    "how large language models work",
    "getting started with machine learning",
    "cybersecurity for small businesses",
    "the future of self driving cars",
    "quantum computing explained simply",
    "how the internet actually works",
    "startup founder stories and lessons",
    "how to start investing in index funds",
    "personal finance for beginners",
    "real estate investing for first timers",
    "understanding inflation and interest rates",
    "how supply chains broke and recovered",
    "negotiating a raise at work",
    "quitting your job to start a business",
    "intermittent fasting evidence",
    "how to sleep better at night",
    "strength training for beginners",
    "marathon training tips for first race",
    "managing anxiety without medication",
    "the science of longevity",
    "gut health and the microbiome",
    "meditation for people who cannot sit still",
    "stand up comedians talking about their craft",
    "funny stories about parenting toddlers",
    "improv comedy behind the scenes",
    "football tactics explained",
    "why the lakers dynasty ended",
    "formula one race strategy",
    "the economics of soccer transfers",
    "olympic athletes training routines",
    "climate change solutions that scale",
    "how electric grids handle renewables",
    "space telescopes and new discoveries",
    "is there life on europa",
    "the physics of black holes",
    "psychology of habit formation",
    "why we procrastinate and how to stop",
    "attachment styles in relationships",
    "raising resilient kids",
    "screen time and child development",
    "learning spanish as an adult",
    "how polyglots learn languages fast",
    "the history of jazz music",
    "how hit songs get written",
    "film directors discussing their movies",
    "oscar winning screenplays breakdown",
    "book recommendations for long flights",
    "classic novels worth rereading",
    "stoic philosophy for modern life",
    "the trolley problem and ethics",
    "how elections are actually run",
    "gerrymandering explained",
    "supreme court landmark decisions",
    "geopolitics of semiconductor chips",
    "war in ukraine analysis",
    "middle east peace process history",
    "how vaccines are developed and tested",
    "nutrition myths debunked by scientists",
    "the story of theranos",
    "enron and corporate fraud",
    "how casinos make money",
    "the rise and fall of blockbuster",
    "video game industry crunch culture",
    "esports becoming mainstream",
    "cooking techniques every beginner should know",
    "the history of coffee",
    "wine tasting for beginners",
    "van life and remote work travel",
    "cheap travel hacks for europe",
    "national parks worth visiting",
    "true stories of survival at sea",
    "mountaineering disasters on everest",
    "the psychology of conspiracy theories",
    "how misinformation spreads online",
]

# Curated queries with structured constraints: (text, filters).
FILTERED = [
    ("short daily news briefing", {"max_duration_seconds": 1200}),
    ("morning news roundup under 20 minutes", {"max_duration_seconds": 1200}),
    ("quick tech news update", {"max_duration_seconds": 900}),
    ("news analysis published this week", {"published_after": "2026-08-21"}),
    ("true crime series published this year", {"published_after": "2026-01-01"}),
    ("recent episodes about the housing market", {"published_after": "2026-06-01"}),
    ("latest ai developments this month", {"published_after": "2026-08-01"}),
    ("recent interviews with startup founders", {"published_after": "2026-05-01"}),
    ("practical uses of artificial intelligence under 45 minutes published this year",
     {"max_duration_seconds": 2700, "published_after": "2026-01-01", "no_explicit": True}),
    ("history episodes under half an hour", {"max_duration_seconds": 1800}),
    ("science explainers under 30 minutes", {"max_duration_seconds": 1800}),
    ("long form interviews over political topics", {}),
    ("kid friendly science episodes", {"no_explicit": True}),
    ("clean comedy without swearing", {"no_explicit": True}),
    ("family friendly history storytelling", {"no_explicit": True}),
    ("bedtime stories for kids", {"no_explicit": True}),
    ("french language news", {"language": "fr"}),
    ("actualites politiques francaises", {"language": "fr"}),
    ("noticias de economia en espanol", {"language": "es"}),
    ("podcasts de historia en espanol", {"language": "es"}),
    ("deutsche nachrichten des tages", {"language": "de"}),
    ("wissenschaft podcast auf deutsch", {"language": "de"}),
    ("noticias de futebol em portugues", {"language": "pt"}),
    ("investing basics under 30 minutes", {"max_duration_seconds": 1800}),
    ("guided meditation under 15 minutes", {"max_duration_seconds": 900}),
    ("workout motivation under 20 minutes", {"max_duration_seconds": 1200}),
    ("recent movie reviews", {"published_after": "2026-07-01"}),
    ("new music releases discussed recently", {"published_after": "2026-07-01"}),
    ("recent health research explained", {"published_after": "2026-04-01"}),
    ("short daily spanish lesson", {"max_duration_seconds": 900, "language": "es"}),
    ("election coverage published this summer", {"published_after": "2026-06-01"}),
    ("clean true crime for road trips", {"no_explicit": True}),
    ("short business news update this week", {"max_duration_seconds": 1200,
                                              "published_after": "2026-08-21"}),
    ("deep dive interviews about climate published this year",
     {"published_after": "2026-01-01"}),
    ("recent episodes about interest rates under 40 minutes",
     {"max_duration_seconds": 2400, "published_after": "2026-06-01"}),
]


def normalize(text: str) -> str:
    return _WORD_RE.sub(" ", text.lower()).strip()


def query_id(text: str, query_type: str) -> str:
    return "q" + hashlib.sha256(f"{query_type}:{text}".encode()).hexdigest()[:8]


def _row(text: str, query_type: QueryType, source: str, filters: dict | None = None) -> dict:
    filters = filters or {}
    return {
        "query_id": query_id(text, query_type.value),
        "query_text": text,
        "query_type": query_type.value,
        "language": filters.get("language"),
        "max_duration_seconds": filters.get("max_duration_seconds"),
        "published_after": filters.get("published_after"),
        "no_explicit": bool(filters.get("no_explicit", False)),
        "source": source,
    }


def build_queries(snapshot_dir: Path) -> pd.DataFrame:
    episodes = pd.read_parquet(snapshot_dir / "episodes.parquet", columns=["id", "podcast_id"])
    podcasts = pd.read_parquet(snapshot_dir / "podcasts.parquet")
    counts = episodes["podcast_id"].value_counts()
    podcasts = podcasts.assign(episodes=podcasts["id"].map(counts).fillna(0).astype(int))

    rows: list[dict] = []

    # Navigational: the biggest English-language shows, queried by name.
    english = podcasts[podcasts["language"].str.startswith("en", na=False)]
    top = english.sort_values(["episodes", "id"], ascending=[False, True])
    seen: set[str] = set()
    for row in top.itertuples():
        name = normalize(str(row.title))
        if len(name) < 3 or name in seen:
            continue
        seen.add(name)
        rows.append(_row(name, QueryType.NAVIGATIONAL, "generated"))
        if sum(r["query_type"] == "navigational" for r in rows) >= NAVIGATIONAL_COUNT:
            break

    # Entity: person-like publisher names (hosts publishing under their own
    # name), by catalog size.
    people = english[english["publisher"].map(lambda p: _looks_like_person(str(p)))]
    seen_publishers: set[str] = set()
    for row in people.sort_values(["episodes", "id"], ascending=[False, True]).itertuples():
        name = normalize(str(row.publisher))
        if name in seen_publishers or name in seen:
            continue
        seen_publishers.add(name)
        rows.append(_row(name, QueryType.ENTITY, "generated"))
        if len(seen_publishers) >= ENTITY_COUNT:
            break

    for text in EXPLORATORY:
        rows.append(_row(text, QueryType.EXPLORATORY, "curated"))
    for text, filters in FILTERED:
        rows.append(_row(text, QueryType.FILTERED, "curated", filters))

    df = pd.DataFrame(rows).sort_values(["query_type", "query_id"], ignore_index=True)
    if df["query_id"].duplicated().any():
        raise ValueError("duplicate query ids; adjust colliding query texts")
    return QueryModel.validate(df)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in df.to_dict("records"):
            clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            f.write(json.dumps(clean, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default="data/snapshots")
    parser.add_argument("--out", default="data/relevance/queries.jsonl")
    args = parser.parse_args(argv)

    df = build_queries(snapshot_mod.resolve(args.snapshot))
    write_jsonl(df, Path(args.out))
    by_type = df["query_type"].value_counts().to_dict()
    print(json.dumps({"queries": len(df), "by_type": by_type, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
