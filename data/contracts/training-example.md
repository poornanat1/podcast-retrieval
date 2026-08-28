# Training-example contract

Human-readable copy of the normalized training-example row shape shared by
all three datasets (search retrieval, personalized retrieval, similar
episodes). The enforced source of truth is `ml/datasets/contracts.py`
(`TrainingExampleModel`); validate any serialized dataset with:

```sh
uv run python -m ml.datasets.validate examples.parquet
```

## Fields

| Field | Type | Rules |
| --- | --- | --- |
| `example_id` | string | non-empty, unique within a dataset |
| `event_time` | timestamp (UTC) | when the labeling interaction happened; never in the future; drives time-based splits |
| `query_text` | string | normalized query; empty for personalized-retrieval examples without a typed query |
| `query_audio_variant` | string | identifier of the speech recording variant the query came from; empty for text-only examples |
| `user_history` | list of episode ids | listener history before `event_time`; must not contain the positive |
| `positive_episode_id` | episode id | the relevant episode; > 0 |
| `negative_episode_ids` | list of episode ids | sampled negatives; must not contain the positive; may be empty (in-batch negatives) |
| `label_source` | enum | see below |
| `label_strength` | float in [0, 1] | label confidence/weight |
| `dataset_version` | string | version tag of the dataset snapshot the row belongs to |

No extra columns are allowed.

## Label sources and reporting classes

Dataset reports must separate real, human, weak, and synthetic labels; each
source maps to exactly one class:

| `label_source` | Class | Meaning |
| --- | --- | --- |
| `search_play` | real | search followed by playback |
| `high_completion` | real | high completion rate (strong positive) |
| `like` | real | like (strong positive) |
| `save` | real | save (strong positive) |
| `impression_no_play` | real | impression without playback (weak negative) |
| `early_skip` | real | early skip (negative) |
| `human_judgment` | human | human-reviewed relevance judgment |
| `topic_similarity` | weak | topic/category similarity (weak positive) |
| `random_negative` | weak | random catalog sample (easy negative) |
| `synthetic_query` | synthetic | generated query for an episode |

## Validation stages

1. **Schema** — types, nullability, uniqueness, id-list well-formedness,
   positive/negative disjointness, no target leakage into history, no future
   timestamps.
2. **Distribution** — label composition (share per class), example and
   distinct-positive counts, negatives-per-example, empty-query share; each
   bound is configurable per dataset version (`ValidationConfig`), with
   bootstrap-friendly defaults that are tightened as real interactions
   accumulate.
