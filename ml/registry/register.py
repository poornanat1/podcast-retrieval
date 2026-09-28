"""Acceptance gates and MLflow model registration for a retriever.

    uv run python -m ml.registry.register --summaries <compare json> \
        --candidate two-tower --baseline vector-e5-small \
        --artifact data/models/<run> [--register]

Gates, evaluated on one comparison run so both systems share a judgment
pool and catalog state:

- **primary**: the candidate is at least as good as the baseline on every
  primary metric (default ``ndcg_at_10`` and ``recall_at_100``);
- **no cohort regression**: on every query type (NDCG@10) and every
  relevant-item cohort (Recall@100) the candidate is not below the
  baseline by more than ``--cohort-tolerance`` (default 0.05 absolute).

``--register`` records the artifact as a version of the registered model
(default ``podfind-two-tower``) whose ``source`` is the training run's
artifact root, tags it with every gate check and the compared metrics,
and sets the alias ``champion`` if all gates passed or ``candidate``
otherwise. The verdict is printed as JSON either way.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

DEFAULT_PRIMARY = ("ndcg_at_10", "recall_at_100")
DEFAULT_MODEL_NAME = "podfind-two-tower"


@dataclasses.dataclass
class GateResult:
    passed: bool
    checks: list[dict]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


def evaluate_gates(
    candidate: dict, baseline: dict, primary: tuple[str, ...] = DEFAULT_PRIMARY,
    cohort_tolerance: float = 0.05,
) -> GateResult:
    """``candidate``/``baseline`` are compare summaries (``global`` and
    ``by_query_type``)."""
    checks: list[dict] = []
    for metric in primary:
        c, b = candidate["global"].get(metric), baseline["global"].get(metric)
        checks.append({"gate": "primary", "metric": metric, "candidate": c, "baseline": b,
                       "passed": c is not None and b is not None and c >= b})
    for query_type in sorted(set(candidate["by_query_type"]) | set(baseline["by_query_type"])):
        c = candidate["by_query_type"].get(query_type, {}).get("ndcg_at_10")
        b = baseline["by_query_type"].get(query_type, {}).get("ndcg_at_10")
        checks.append({"gate": "cohort", "metric": f"{query_type}__ndcg_at_10", "candidate": c,
                       "baseline": b, "passed": c is not None and b is not None
                       and c >= b - cohort_tolerance})
    cohorts = sorted(k for k in baseline["global"] if k.startswith("recall_at_100__")
                     and not k.endswith("__queries"))
    for key in cohorts:
        c, b = candidate["global"].get(key), baseline["global"].get(key)
        checks.append({"gate": "cohort", "metric": key, "candidate": c, "baseline": b,
                       "passed": c is not None and b is not None and c >= b - cohort_tolerance})
    return GateResult(passed=all(ch["passed"] for ch in checks), checks=checks)


def register(
    artifact: Path, gates: GateResult, candidate: dict, baseline: dict,
    tracking_uri: str, model_name: str = DEFAULT_MODEL_NAME, run_id: str | None = None,
) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    meta = json.loads((artifact / "artifact.json").read_text())
    run_id = run_id or meta.get("mlflow_run_id")
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            model_name, description="Two-tower first-stage retriever over the episode catalog."
        )
    run = client.get_run(run_id)
    source = run.info.artifact_uri
    tags = {
        "artifact_dir": str(artifact),
        "dataset_version": str(meta.get("dataset_version")),
        "feature_contract_sha256": str(meta.get("feature_contract_sha256")),
        "git_commit": str(meta.get("git_commit")),
        "gates_passed": str(gates.passed).lower(),
        "candidate_system": candidate["system"],
        "baseline_system": baseline["system"],
        "candidate_ndcg_at_10": str(candidate["global"].get("ndcg_at_10")),
        "baseline_ndcg_at_10": str(baseline["global"].get("ndcg_at_10")),
        "candidate_recall_at_100": str(candidate["global"].get("recall_at_100")),
        "baseline_recall_at_100": str(baseline["global"].get("recall_at_100")),
    }
    version = client.create_model_version(
        model_name, source=source, run_id=run_id, tags=tags,
        description=json.dumps(gates.to_dict()),
    )
    alias = "champion" if gates.passed else "candidate"
    client.set_registered_model_alias(model_name, alias, version.version)
    return {"model": model_name, "version": version.version, "alias": alias, "source": source}


def main(argv: list[str] | None = None) -> int:
    from ml.evaluation.run import DEFAULT_MLFLOW_URI

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summaries", required=True, help="JSON from ml.evaluation.compare")
    parser.add_argument("--candidate", required=True, help="candidate system name")
    parser.add_argument("--baseline", required=True, help="baseline system name")
    parser.add_argument("--primary", default=",".join(DEFAULT_PRIMARY))
    parser.add_argument("--cohort-tolerance", type=float, default=0.05)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--run-id", default="", help="training run on the tracking server")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)

    summaries = {s["system"]: s for s in json.loads(Path(args.summaries).read_text())}
    candidate, baseline = summaries[args.candidate], summaries[args.baseline]
    gates = evaluate_gates(candidate, baseline, tuple(args.primary.split(",")),
                           args.cohort_tolerance)
    verdict = {"candidate": args.candidate, "baseline": args.baseline, **gates.to_dict()}
    if args.register:
        if not args.artifact:
            parser.error("--register needs --artifact")
        verdict["registered"] = register(
            Path(args.artifact), gates, candidate, baseline,
            os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI), args.model_name,
            args.run_id or None,
        )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
