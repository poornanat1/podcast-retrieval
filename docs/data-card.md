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
2. **Publisher transcripts** — plain-text/SRT/VTT/JSON transcripts linked
   from feeds (~9% episode coverage at the current catalog size).
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
| synthetic | synthetic_query | generated pairs, clearly flagged |

Current bootstrap training datasets are 100% weak + synthetic; this is
expected pre-launch and reported honestly in every manifest.

## Evaluation ground truth

The relevance set (`data/relevance/`) contains ~180 queries and ~3.4k graded
judgments over pooled lexical candidates. Judgments are currently
LLM-generated (`judge: llm:<model>`), each with a rationale; human review
via the grading CLI overrides LLM grades at export. The eval set is **never
used for training** — its only role is measurement.

## Known biases and limitations

- **Popularity skew:** the catalog was seeded from trending charts plus
  topical searches, over-representing head shows.
- **Language skew:** ~69% English; es/fr/de/pt at ~100 podcasts each.
- **Transcript skew:** transcript-bearing episodes concentrate in large,
  professionally produced shows (~17% of podcasts).
- **Synthetic query gap:** title-derived queries are cleaner and more
  entity-like than real voice queries; speech-noise robustness is not
  represented in bootstrap data.
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
