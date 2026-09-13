# Relevance Judgments: Human Review & LLM Annotation

Ground truth for evaluating retrieval quality comes from human-labeled relevance judgments.

## Relevance Grading Scale

| Grade | Definition | Example |
|-------|-----------|---------|
| **Relevant** | Episode answers the query well | Query: "machine learning in healthcare"; Episode: 45-min discussion of ML applications in medical imaging |
| **Somewhat Relevant** | Episode has useful info but is partial or tangential | Query: "machine learning in healthcare"; Episode: Mentions ML once in a 2-hour general tech show |
| **Not Relevant** | Episode doesn't address the query | Query: "machine learning in healthcare"; Episode: Cooking podcast (no ML content) |

## Relevance Pooling: Creating the Judgment Set

**Goal:** Assemble a diverse set of {query, episode} pairs for human review.

### Process

1. **Select queries**: ~2,500 representative search queries
   - From user logs (if available)
   - Manually curated (diverse intent, difficulty)
   - Topics: AI, business, science, culture, etc.

2. **Pool candidates**: Run multiple retrieval systems on each query
   - Lexical (BM25)
   - Vector (multilingual-e5-small)
   - Hybrid (RRF)
   - Popularity baselines

3. **Deduplicate**: Keep top-K unique episodes across all systems
   - Reduces overlap; ensures diverse coverage
   - Typical pool size: 5-10 episodes per query

4. **Annotate**: Human judges grade {query, episode} pair

### Pooling Configuration

```json
{
  "name": "relevance-pool-v1",
  "queries": "experiments/eval/queries.json",
  "pool_systems": [
    "lexical-fts",
    "vector-e5-small",
    "hybrid-rrf",
    "popularity-global",
    "popularity-category"
  ],
  "pool_size": 10,
  "judges_per_pair": 1
}
```

**Run pooling:**
```bash
make relevance-pool POOL_SYSTEMS=lexical-fts,vector-e5-small,hybrid-rrf
```

**Output:**
```
data/relevance/
├── relevance-pool-v1.parquet  # {query, episode_id, search_system, rank}
└── metadata.json
```

## Human Review Workflow

### Setup

1. **Annotator interface**: Web UI (or spreadsheet for small sets)
2. **Instructions**: Clear rubric + examples
3. **Batch size**: 100-200 pairs per annotator per session
4. **Inter-rater agreement**: 2-3 judges per pair; resolve disagreements

### Judging Interface

```
Query: "Find episodes about practical AI applications"

Episode: "AI in Healthcare" (Dr. Sarah Johnson) | 45 min
Title: How Machine Learning is Transforming Medical Diagnostics
Description: Discussion of ML applications in radiology and pathology...

[ Relevant ]  [ Somewhat Relevant ]  [ Not Relevant ]  [ Skip ]

Grade: [Selected: Relevant]
Notes: Clear discussion of practical AI use cases; well-aligned with query.
```

### Resolution of Disagreements

If multiple judges disagree:
- **2/3 relevant** → relevant
- **1/3 relevant** → somewhat relevant
- **0/3 relevant** → not relevant

## LLM-Assisted Annotation (Future)

For scaling beyond human budget, use LLM to pre-annotate then review:

```python
# ml/relevance/llm_judge.py
from openai import OpenAI

client = OpenAI()

def llm_judge_relevance(query: str, episode: dict) -> str:
    """Grade relevance of episode to query using Claude."""
    
    prompt = f"""
    Query: {query}
    
    Episode Title: {episode['title']}
    Episode Description: {episode['description']}
    Episode Transcript (first 500 words): {episode['transcript'][:500]}
    
    Grade the relevance of this episode to the query:
    - "relevant": Episode answers the query well
    - "somewhat_relevant": Episode has useful info but is partial
    - "not_relevant": Episode doesn't address the query
    
    Return only the grade (one word).
    """
    
    response = client.messages.create(
        model="claude-opus",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.content[0].text.strip().lower()

# Batch judge
for query, episode in pool:
    grade = llm_judge_relevance(query, episode)
    # Store in database
```

**Cost estimate:** At $0.003 per 1K tokens, ~2.5M tokens for 2,500 queries × 10 episodes = ~$7.50

**Accuracy:** LLM-grades typically align with human judgment 75-85% of the time.

## Storage & Schema

### Relevance Database

```sql
CREATE TABLE relevance_judgments (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    query TEXT NOT NULL,
    episode_id BIGINT NOT NULL REFERENCES episodes(id),
    relevance_grade VARCHAR(20) NOT NULL,  -- 'relevant', 'somewhat_relevant', 'not_relevant'
    
    -- Metadata
    annotator_id VARCHAR(100),  -- Who judged
    judged_at TIMESTAMP DEFAULT NOW(),
    notes TEXT,  -- Annotator comments
    
    UNIQUE(query, episode_id, annotator_id)
);
```

### Relevance Parquet Export

```python
# ml/relevance/export.py
import pandas as pd
import psycopg

conn = psycopg.connect("postgres://...")
judgments = pd.read_sql(
    "SELECT query, episode_id, relevance_grade FROM relevance_judgments",
    conn
)

# Export to Parquet
judgments.to_parquet("data/relevance/relevance-pool-v1.parquet", index=False)
```

## Analytics on Judgments

### Coverage

What % of episodes have judgments?

```python
total_episodes = 540000
judged_episodes = df['episode_id'].nunique()
coverage = judged_episodes / total_episodes
# e.g., 0.04 (4% of catalog judged)
```

### Query Difficulty

How many relevant episodes per query?

```python
relevant_per_query = df[df['relevance_grade'] == 'relevant'].groupby('query').size()
easy_queries = relevant_per_query[relevant_per_query > 5]  # Many relevant episodes
hard_queries = relevant_per_query[relevant_per_query == 1]  # Only 1 relevant
```

### Bias Detection

Does any system dominate the pool?

```python
# Pool was constructed from multiple systems
# Do lexical results skew the pool? Check diversity

from collections import Counter
episode_source = Counter(df['sourced_by_system'])
# Should be balanced across systems
```

## Updating Judgments

### Adding New Queries

1. Identify new queries (user logs, brainstorm)
2. Run pooling on new queries only
3. Human judge the new pool
4. Merge with existing judgments

```bash
# Append to existing dataset
new_judgments = pd.read_parquet("data/relevance/new-pool.parquet")
existing = pd.read_parquet("data/relevance/relevance-pool-v1.parquet")
combined = pd.concat([existing, new_judgments]).drop_duplicates()
combined.to_parquet("data/relevance/relevance-pool-v2.parquet")
```

### Re-judging (Quality Check)

Periodically re-judge a random sample:
- 5-10% of existing judgments
- Different annotators
- Check agreement; if <75%, review and clarify rubric

## Integration with Evaluation

See [07-evaluation.md](07-evaluation.md) for how judgments are used in metrics.

**Key idea:** Judgments provide ground truth; metrics compute how well retrievers rank episodes relative to that ground truth.

## Best Practices

1. **Clear rubric**: Define relevance precisely with examples
2. **Calibration**: Have judges review examples + rubric before annotation
3. **Batch review**: Have a senior person sample-check 5-10% of annotations
4. **Document disagreements**: If judges disagree, capture discussion
5. **Version:** Judgments are data; versioned like snapshots/datasets
6. **Cleanup**: Remove judgments if episode is removed from catalog

## Future Work

1. **Dynamic pooling**: Update pool based on retriever improvements (avoid stale judgments)
2. **Active learning**: Prioritize judging episodes where retrievers disagree
3. **Aspect-based relevance**: Grade separately for {semantic match, currency, authority}
4. **Fine-grained labels**: Numeric relevance (0-5) instead of categorical
5. **Crowd-source**: Crowdworkers via Amazon Mechanical Turk or Upwork
