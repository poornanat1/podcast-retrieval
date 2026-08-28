"""Schema and distribution validation for training-example datasets.

Usage as a pipeline stage::

    from ml.datasets.validate import ValidationConfig, validate_examples
    result = validate_examples(df, ValidationConfig())
    if not result.ok:
        raise SystemExit(result.summary())

Usage from the shell::

    uv run python -m ml.datasets.validate examples.parquet --min-examples 1000

The schema stage enforces the typed contract in ``ml.datasets.contracts``.
The distribution stage reports label composition — real, human, weak, and
synthetic labels must be separable — and fails the dataset when composition
or coverage falls outside the configured bounds.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from ml.datasets.contracts import LABEL_CLASS, LabelClass, LabelSource, TrainingExampleModel

_MAX_REPORTED_ERRORS = 20


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    """Bounds for the distribution stage. Defaults are permissive enough for
    the bootstrap phase (weak/synthetic-heavy) and are tightened per dataset
    version as real interactions accumulate."""

    min_examples: int = 100
    min_distinct_positives: int = 10
    max_weak_share: float = 1.0
    max_synthetic_share: float = 1.0
    min_real_share: float = 0.0
    min_human_share: float = 0.0
    max_empty_query_share: float = 1.0
    min_negatives_per_example: int = 0


@dataclasses.dataclass
class DistributionReport:
    examples: int = 0
    distinct_positives: int = 0
    distinct_queries: int = 0
    empty_query_share: float = 0.0
    negatives_min: int = 0
    negatives_mean: float = 0.0
    negatives_max: int = 0
    event_time_min: str = ""
    event_time_max: str = ""
    source_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    class_shares: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ValidationResult:
    rows: int
    schema_errors: list[str]
    distribution_failures: list[str]
    report: DistributionReport | None

    @property
    def ok(self) -> bool:
        return not self.schema_errors and not self.distribution_failures

    def summary(self) -> str:
        payload: dict[str, Any] = {
            "rows": self.rows,
            "ok": self.ok,
            "schema_errors": self.schema_errors,
            "distribution_failures": self.distribution_failures,
        }
        if self.report is not None:
            payload["report"] = dataclasses.asdict(self.report)
        return json.dumps(payload, indent=2)


def load_examples(path: str | Path) -> pd.DataFrame:
    """Load a serialized dataset (.parquet or .jsonl) into a DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"unsupported dataset format {path.suffix!r} for {path}")
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def build_report(df: pd.DataFrame) -> DistributionReport:
    """Compute the distribution report for a schema-valid DataFrame."""
    negative_counts = df["negative_episode_ids"].map(len)
    queries = df["query_text"].fillna("")
    counts = df["label_source"].value_counts()
    by_class: dict[str, int] = {c.value: 0 for c in LabelClass}
    for source, count in counts.items():
        by_class[LABEL_CLASS[LabelSource(source)].value] += int(count)
    total = len(df)
    return DistributionReport(
        examples=total,
        distinct_positives=int(df["positive_episode_id"].nunique()),
        distinct_queries=int(queries[queries != ""].nunique()),
        empty_query_share=float((queries == "").mean()) if total else 0.0,
        negatives_min=int(negative_counts.min()) if total else 0,
        negatives_mean=round(float(negative_counts.mean()), 2) if total else 0.0,
        negatives_max=int(negative_counts.max()) if total else 0,
        event_time_min=str(df["event_time"].min()) if total else "",
        event_time_max=str(df["event_time"].max()) if total else "",
        source_counts={str(k): int(v) for k, v in counts.items()},
        class_shares={
            c: round(n / total, 4) if total else 0.0 for c, n in by_class.items()
        },
    )


def check_distribution(report: DistributionReport, config: ValidationConfig) -> list[str]:
    failures: list[str] = []
    if report.examples < config.min_examples:
        failures.append(f"examples {report.examples} < min_examples {config.min_examples}")
    if report.distinct_positives < config.min_distinct_positives:
        failures.append(
            f"distinct positives {report.distinct_positives} < "
            f"min_distinct_positives {config.min_distinct_positives}"
        )
    shares = report.class_shares
    if shares.get(LabelClass.WEAK.value, 0.0) > config.max_weak_share:
        failures.append(f"weak share {shares[LabelClass.WEAK.value]} > {config.max_weak_share}")
    if shares.get(LabelClass.SYNTHETIC.value, 0.0) > config.max_synthetic_share:
        failures.append(
            f"synthetic share {shares[LabelClass.SYNTHETIC.value]} > {config.max_synthetic_share}"
        )
    if shares.get(LabelClass.REAL.value, 0.0) < config.min_real_share:
        failures.append(f"real share {shares[LabelClass.REAL.value]} < {config.min_real_share}")
    if shares.get(LabelClass.HUMAN.value, 0.0) < config.min_human_share:
        failures.append(f"human share {shares[LabelClass.HUMAN.value]} < {config.min_human_share}")
    if report.empty_query_share > config.max_empty_query_share:
        failures.append(
            f"empty-query share {report.empty_query_share} > {config.max_empty_query_share}"
        )
    if report.negatives_min < config.min_negatives_per_example:
        failures.append(
            f"min negatives per example {report.negatives_min} < "
            f"{config.min_negatives_per_example}"
        )
    return failures


def validate_examples(
    df: pd.DataFrame, config: ValidationConfig | None = None
) -> ValidationResult:
    """Run the schema stage, then (only when it passes) the distribution stage."""
    config = config or ValidationConfig()
    try:
        validated = TrainingExampleModel.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        errors = [
            f"{row.check}: column={row.column} case={row.failure_case!r}"
            for row in cases.head(_MAX_REPORTED_ERRORS).itertuples()
        ]
        if len(cases) > _MAX_REPORTED_ERRORS:
            errors.append(f"... {len(cases) - _MAX_REPORTED_ERRORS} more failure cases")
        return ValidationResult(rows=len(df), schema_errors=errors,
                                distribution_failures=[], report=None)

    report = build_report(validated)
    return ValidationResult(
        rows=len(validated),
        schema_errors=[],
        distribution_failures=check_distribution(report, config),
        report=report,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="dataset file (.parquet or .jsonl)")
    cfg = ValidationConfig()
    parser.add_argument("--min-examples", type=int, default=cfg.min_examples)
    parser.add_argument("--min-distinct-positives", type=int, default=cfg.min_distinct_positives)
    parser.add_argument("--max-weak-share", type=float, default=cfg.max_weak_share)
    parser.add_argument("--max-synthetic-share", type=float, default=cfg.max_synthetic_share)
    parser.add_argument("--min-real-share", type=float, default=cfg.min_real_share)
    parser.add_argument("--min-human-share", type=float, default=cfg.min_human_share)
    parser.add_argument("--max-empty-query-share", type=float, default=cfg.max_empty_query_share)
    parser.add_argument("--min-negatives", type=int, default=cfg.min_negatives_per_example)
    args = parser.parse_args(argv)

    result = validate_examples(
        load_examples(args.path),
        ValidationConfig(
            min_examples=args.min_examples,
            min_distinct_positives=args.min_distinct_positives,
            max_weak_share=args.max_weak_share,
            max_synthetic_share=args.max_synthetic_share,
            min_real_share=args.min_real_share,
            min_human_share=args.min_human_share,
            max_empty_query_share=args.max_empty_query_share,
            min_negatives_per_example=args.min_negatives,
        ),
    )
    print(result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
