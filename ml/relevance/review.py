"""Interactive grading of pooled judgment tasks.

    uv run python -m ml.relevance.review

Walks every unjudged (query, episode) pair grouped by query, shows the
episode in context, and records graded judgments append-only to the
judgments file. Grades: 0 irrelevant, 1 marginal, 2 relevant, 3 highly
relevant; s skips the pair, S skips the rest of the query, q quits (progress
is saved after every judgment).
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from ml.relevance.contracts import GRADES


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="data/relevance/tasks.jsonl")
    parser.add_argument("--judgments", default="data/relevance/judgments.jsonl")
    parser.add_argument("--judge", default=getpass.getuser())
    args = parser.parse_args(argv)

    tasks = load_jsonl(Path(args.tasks))
    if not tasks:
        print(f"no tasks at {args.tasks}; run `make relevance-pool` first")
        return 1
    judgments_path = Path(args.judgments)
    done = {(j["query_id"], j["episode_id"]) for j in load_jsonl(judgments_path)}

    pending = [t for t in tasks if (t["query_id"], t["episode_id"]) not in done]
    print(f"{len(pending)} of {len(tasks)} pairs unjudged\n")
    print("grades: " + "; ".join(f"{g}={desc.split(':')[0]}" for g, desc in GRADES.items()))
    print("keys: 0-3 grade | s skip pair | S skip query | q quit\n")

    judged = 0
    with judgments_path.open("a") as out:
        for _, group in groupby(pending, key=lambda t: t["query_id"]):
            group = list(group)
            head = group[0]
            filters = ", ".join(
                f"{k}={head[k]}" for k in ("language",) if head.get(k)
            )
            print("=" * 72)
            print(f"QUERY [{head['query_type']}]: {head['query_text']}"
                  + (f"  ({filters})" if filters else ""))
            skip_query = False
            for task in group:
                if skip_query:
                    break
                duration = task.get("duration_seconds")
                minutes = f"{duration // 60}m" if duration else "?"
                print(f"\n  #{task['rank']} [{task['episode_id']}] "
                      f"{task['podcast_title']} — {task['episode_title']}")
                print(f"     {minutes} | {task.get('language', '')} | "
                      f"published {str(task.get('published_at', ''))[:10]} | "
                      f"transcript={'yes' if task.get('has_transcript') else 'no'}")
                if task.get("description"):
                    print(f"     {task['description'][:200]}")
                while True:
                    choice = input("  grade> ").strip()
                    if choice == "q":
                        print(f"\nsaved {judged} judgments")
                        return 0
                    if choice == "s":
                        break
                    if choice == "S":
                        skip_query = True
                        break
                    if choice in {"0", "1", "2", "3"}:
                        out.write(json.dumps({
                            "query_id": task["query_id"],
                            "episode_id": task["episode_id"],
                            "grade": int(choice),
                            "judge": args.judge,
                            "judged_at": datetime.now(UTC).isoformat(),
                            "notes": "",
                        }, sort_keys=True) + "\n")
                        out.flush()
                        judged += 1
                        break
                    print("  enter 0-3, s, S, or q")
    print(f"\nall pairs judged; saved {judged} this session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
