"""Copy an MLflow run from one tracking store to another.

    uv run python -m ml.training.sync_run --run-id <id> \
        --source sqlite:///path/to/mlflow.db --target http://localhost:5001

Lets a training job log to a local SQLite store (no tracking server, and
on a memory-starved laptop no Docker VM, running alongside it) and publish
the run — params, tags, full metric histories, and artifacts — to the
shared server afterwards. The copy gets the same experiment name and a
``source_run_id`` tag pointing at the original.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


def sync_run(run_id: str, source_uri: str, target_uri: str) -> str:
    source = MlflowClient(tracking_uri=source_uri)
    target = MlflowClient(tracking_uri=target_uri)
    run = source.get_run(run_id)
    experiment_name = source.get_experiment(run.info.experiment_id).name
    experiment = target.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else target.create_experiment(
        experiment_name
    )

    copied = target.create_run(
        experiment_id,
        start_time=run.info.start_time,
        tags={**run.data.tags, "source_run_id": run_id, "source_tracking_uri": source_uri},
        run_name=run.info.run_name,
    )
    copied_id = copied.info.run_id
    for key, value in run.data.params.items():
        target.log_param(copied_id, key, value)

    metrics: list[Metric] = []
    for key in run.data.metrics:
        metrics.extend(
            Metric(key, m.value, m.timestamp, m.step)
            for m in source.get_metric_history(run_id, key)
        )
    for start in range(0, len(metrics), 1000):  # server batch limit
        target.log_batch(copied_id, metrics=metrics[start:start + 1000])

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=source_uri, dst_path=tmp
        ))
        if any(local.iterdir()):
            target.log_artifacts(copied_id, str(local))

    target.set_terminated(copied_id, status=run.info.status, end_time=run.info.end_time)
    return copied_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", required=True, help="source tracking URI")
    parser.add_argument("--target", required=True, help="target tracking URI")
    args = parser.parse_args(argv)
    copied = sync_run(args.run_id, args.source, args.target)
    print(f"copied run {args.run_id} -> {copied} on {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
