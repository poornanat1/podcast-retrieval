# Data card — PodFind training and evaluation data

**Status: draft.** Updated as datasets evolve; numbers cite the manifest of
the dataset version they describe.

## What the data is

PodFind builds datasets for podcast search and recommendation from three
sources:

1. **Catalog metadata** — podcast and episode titles, descriptions,
   categories, publishers, languages, durations, and publication dates,
   ingested from publisher RSS feeds discovered via the Particle data
   platform. Audio is never downloaded; only enclosure URLs are stored.
   **Corpus freeze:** feed polling and discovery were stopped on
   2026-09-13 with the catalog at 1,788 podcasts and 795,533 episodes, so
   every retrieval system is evaluated against the same fixed corpus.
   Judgments and metrics dated after 2026-09-13 assume this frozen state.
   On 2026-09-18 the worker briefly ran again (see
   [09-catalog.md](09-catalog.md)): 1,683 episodes and 2 podcasts were
   added and 8,976 episodes' metadata refreshed, so comparisons from that
   date through 2026-10-18 use a 797,216-episode corpus and re-run every
   baseline on it. On 2026-10-18 the added episodes and podcasts were
   deleted (with their vectors and the 265 relevance judgments on them),
   restoring 795,533 episodes and 1,788 podcasts; the metadata refreshes
   remain.
2. **Publisher transcripts** — plain-text/SRT/VTT/JSON transcripts linked
   from feeds (~11% episode coverage at the current catalog size).
3. **Interaction events** — not yet collected; the schema exists and label
   sources for real interactions are reserved in the taxonomy below.

## Label taxonomy

Every training example carries a `label_source` mapping to exactly one
reporting class; dataset manifests report the class composition, and no
report may blend them silently:

| Class | Sources | Trust |
| --- | --- | --- |
| real | search_play, high_completion, like, save, impression_no_play, early_skip | observed user behavior (none collected yet) |
| human | human_judgment | human-reviewed relevance judgments |
| weak | topic_similarity, random_negative | heuristic supervision |
| synthetic | synthetic_query | generated pairs, clearly flagged: title-derived queries or LLM-written queries |

Current bootstrap training datasets are 100% weak + synthetic; this is
expected pre-launch and reported honestly in every manifest.

## Current training dataset

`search-retrieval-llm-names-1.0.0-56e2c0708f7e`, which trains the
registered champion ([model-card.md](model-card.md)), is 100% synthetic:

| | train | validation | test |
|---|---|---|---|
| rows | 24,348 | 1,087 | 1,408 |

- **Queries**: 18,451 LLM-written natural-language searches, one per
  sampled episode, and 5,897 show-name or publisher-name queries paired
  with episodes of that show (training split). Shows whose title or
  publisher is a relevance-set query are excluded from the name queries.
  Names are lower-cased and stripped to ASCII letters and digits, so
  non-English names lose accented letters (the relevance set's English
  names are normalized the same way).
- **Negatives**: 2 lexical-mined candidates and 6 random catalog
  episodes per example.
- **Splits**: by episode publication time (validation from 2026-06-01,
  test from 2026-07-15).
- **Snapshot**: `baf91376ae98` (2026-08-28; 540,875 episodes, 1,314
  shows). Snapshot `990134867a2f` is derived from it with current
  transcript excerpts added (48,733 episodes) for transcript ablations;
  its rows are otherwise identical.
- **Transcript coverage**: 2,078 of the 24,348 training positives (8.5%)
  have a transcript, against 14% of the negatives, so a transcript input
  has little positive signal to learn from.

**Coverage of the live catalog.** The catalog the models are evaluated on
has grown past the snapshot: of 795,533 episodes, 24,122 were training
positives, 517,000 are other episodes of shows with training positives,
254,411 belong to shows with no training positive, and 254,658 entered
the catalog after the snapshot (the ablation report measured these
slices on the 797,216-episode corpus). Evaluation reports recall on each slice
([experiments/ablation-cold-start-report.md](../experiments/ablation-cold-start-report.md)).

## Evaluation ground truth

The relevance set (`data/relevance/`) contains 180 queries and ~71.5k graded
judgments over candidates pooled from every retrieval system under
comparison. Judgments are currently
LLM-generated (`judge: llm:<model>`), each with a rationale; human review
via the grading CLI overrides LLM grades at export. The eval set is **never
used for training** — its only role is measurement.

## Known biases and limitations

- **Popularity skew:** the catalog was seeded from trending charts plus
  topical searches, over-representing head shows.
- **Language skew:** ~77% English; es/fr/de/pt at ~100 podcasts each.
- **Transcript skew:** transcript-bearing episodes concentrate in large,
  professionally produced shows (~17% of podcasts).
- **Synthetic query gap:** LLM-written queries reflect one model's idea
  of how people search, and name queries label every episode of a show as
  relevant, which teaches show membership rather than episode relevance
  (a documented failure mode of the champion). Earlier title-derived
  queries were cleaner and more entity-like than real searches and are no
  longer used. Speech-noise robustness is not represented.
- **Evaluation language skew:** 172 of the 180 relevance-set queries are
  English; quality in other languages is not yet measurable.
- **LLM judgment risk:** machine-graded relevance may err systematically;
  agreement with human spot-audits should be measured and reported before
  trusting small metric deltas.

## Reproducibility and versioning

Datasets are built by a deterministic pipeline: (config, code revision,
raw-data snapshot) → identical rows, verified by tests. Every dataset
version embeds a content hash; manifests record the git commit, config
hash, snapshot id, split boundaries, and label composition. Published
versions in object storage are immutable.

## Privacy and licensing

- Catalog content remains the property of its publishers; only required
  metadata and transcript text are stored, with attribution preserved and
  feed removal honored (see `docs/licensing.md`).
- No personal user data exists in any current dataset. When interaction
  events arrive, examples will carry pseudonymous ids only, and voice
  recordings are deleted after transcription unless users opt in.
