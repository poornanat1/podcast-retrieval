"""LLM-graded relevance judgments over the pooled tasks.

    uv run python -m ml.relevance.llm_judge [--limit N] [--model gpt-5.1]

Machine judgments are always labeled ``llm:<model>`` in the ``judge`` field —
never as human. A human grading the same (query, episode) pair later via
``make relevance-review`` takes precedence at export time, so spot-auditing
and correcting the LLM is cheap. Progress is append-only and resumable:
already-judged pairs (by any judge) are skipped.

Requires ``OPENAI_API_KEY`` (read from the environment or ``.env``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-5.1"

RUBRIC = """You are grading podcast search results for relevance. For each \
candidate episode, assign a grade:

3 = highly relevant: a near-perfect answer for the query
2 = relevant: satisfies the query, even if not the ideal pick
1 = marginal: topic-adjacent; a user would feel "close, but no"
0 = irrelevant: would not satisfy the query at all

Rules:
- Judge the query's INTENT, not keyword overlap. Candidates were retrieved \
lexically; matching words with a different meaning or context is 0-1.
- navigational queries name a specific show: any episode OF that show is 3; \
episodes of a different but similar show are 0-1, not 2.
- entity queries name a person: episodes BY or substantially FEATURING them \
are 2-3; passing mentions are 1.
- filtered queries: structured constraints (duration, date, language, \
explicit) were already enforced before you see the candidates — grade only \
the topical intent.
- Judge from the metadata shown (show, title, description). When genuinely \
torn between two grades, choose the lower one.
- Give every candidate a grade and a one-sentence rationale."""


class EpisodeGrade(BaseModel):
    episode_id: int
    grade: int = Field(ge=0, le=3)
    rationale: str


class BatchGrades(BaseModel):
    grades: list[EpisodeGrade]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_batch(tasks: list[dict]) -> str:
    head = tasks[0]
    lines = [
        f"QUERY ({head['query_type']}): {head['query_text']}",
        "",
        "CANDIDATES:",
    ]
    for task in tasks:
        duration = task.get("duration_seconds")
        minutes = f"{duration // 60} min" if duration else "unknown length"
        lines.append(
            f"- episode_id={task['episode_id']} | show: {task['podcast_title']} | "
            f"title: {task['episode_title']} | {minutes} | "
            f"published {str(task.get('published_at', ''))[:10]}"
        )
        if task.get("description"):
            lines.append(f"  description: {task['description'][:280]}")
    lines.append("")
    lines.append("Grade every candidate.")
    return "\n".join(lines)


def grade_batch(client: OpenAI, model: str, tasks: list[dict]) -> list[EpisodeGrade]:
    """One API call grading all candidates for a single query."""
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "low"},
        instructions=RUBRIC,
        input=format_batch(tasks),
        text_format=BatchGrades,
    )
    return response.output_parsed.grades


def judge(
    tasks: list[dict],
    judgments_path: Path,
    grader,
    judge_name: str,
    limit: int = 0,
) -> dict:
    """Grade all unjudged pairs, appending results after each query. The
    grader is injected so tests run without the network."""
    done = {(j["query_id"], j["episode_id"]) for j in load_jsonl(judgments_path)}
    pending = [t for t in tasks if (t["query_id"], t["episode_id"]) not in done]

    stats = {"queries": 0, "judged": 0, "missing": 0, "failed_queries": 0}
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    with judgments_path.open("a") as out:
        for _, group in groupby(pending, key=lambda t: t["query_id"]):
            if limit and stats["queries"] >= limit:
                break
            batch = list(group)
            stats["queries"] += 1
            wanted = {t["episode_id"] for t in batch}
            try:
                grades = grader(batch)
            except Exception as exc:  # noqa: BLE001 - one bad query must not kill the run
                print(f"  query {batch[0]['query_id']} failed: {exc}", file=sys.stderr)
                stats["failed_queries"] += 1
                continue

            now = datetime.now(UTC).isoformat()
            returned = set()
            for grade in grades:
                if grade.episode_id not in wanted or grade.episode_id in returned:
                    continue
                returned.add(grade.episode_id)
                out.write(json.dumps({
                    "query_id": batch[0]["query_id"],
                    "episode_id": grade.episode_id,
                    "grade": grade.grade,
                    "judge": judge_name,
                    "judged_at": now,
                    "notes": grade.rationale[:300],
                }, sort_keys=True) + "\n")
                stats["judged"] += 1
            out.flush()
            stats["missing"] += len(wanted - returned)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="data/relevance/tasks.jsonl")
    parser.add_argument("--judgments", default="data/relevance/judgments.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="max queries (0 = all)")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (env or .env)", file=sys.stderr)
        return 1

    tasks = load_jsonl(Path(args.tasks))
    if not tasks:
        print(f"no tasks at {args.tasks}; run `make relevance-pool` first", file=sys.stderr)
        return 1

    client = OpenAI()
    stats = judge(
        tasks,
        Path(args.judgments),
        grader=lambda batch: grade_batch(client, args.model, batch),
        judge_name=f"llm:{args.model}",
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["failed_queries"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
