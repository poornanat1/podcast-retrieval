"""Promote a serving bundle, gated on parity.

    uv run python -m ml.serving.promote --bundle data/serving/bundles/<id>

Refuses unless:

- every bundle file matches its manifest hash (the bundle is unchanged);
- ``parity.json`` exists, passed every check, names this bundle, and was
  computed on the bundle's current ``golden.jsonl`` (same hash).

On success it points ``data/serving/ACTIVE`` at the bundle (the model
server's default) and appends the promotion, with the parity summary, to
``data/serving/promotions.jsonl``. Exit status 1 and no change on refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ml.serving.bundle import verify
from ml.serving.golden import GOLDEN
from ml.serving.parity import PARITY
from ml.serving.runtime import ACTIVE_POINTER


class PromotionRefused(Exception):
    pass


def gate(bundle: Path) -> dict:
    """The parity report, if the bundle may be promoted; raises otherwise."""
    try:
        manifest = verify(bundle)
    except (OSError, ValueError, KeyError) as err:
        raise PromotionRefused(f"bundle does not verify: {err}") from err
    report_path, golden_path = bundle / PARITY, bundle / GOLDEN
    if not report_path.exists():
        raise PromotionRefused("no parity report; run ml.serving.parity first")
    if not golden_path.exists():
        raise PromotionRefused("no golden examples; run ml.serving.golden first")
    report = json.loads(report_path.read_text())
    if report.get("bundle_id") != manifest["bundle_id"]:
        raise PromotionRefused(f"parity report is for bundle {report.get('bundle_id')}, "
                               f"not {manifest['bundle_id']}")
    if report.get("golden_sha256") != hashlib.sha256(golden_path.read_bytes()).hexdigest():
        raise PromotionRefused("parity report was computed on different golden examples")
    failed = [c["name"] for c in report.get("checks", []) if not c["passed"]]
    if failed or not report.get("passed"):
        raise PromotionRefused("parity failed: " + "; ".join(failed or ["report not passed"]))
    return report


def promote(bundle: Path, pointer: Path = ACTIVE_POINTER) -> dict:
    report = gate(bundle)
    previous = pointer.read_text().strip() if pointer.exists() else None
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(report["bundle_id"] + "\n")
    tmp.replace(pointer)
    entry = {"promoted_at": datetime.now(UTC).isoformat(), "bundle_id": report["bundle_id"],
             "previous": previous, "parity_checked_at": report["checked_at"],
             "parity_checks": len(report["checks"])}
    with (pointer.parent / "promotions.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    try:
        entry = promote(Path(args.bundle))
    except PromotionRefused as err:
        print(f"promotion refused: {err}", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
