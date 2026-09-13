# Evaluation: Metrics, Harness & Reproducibility

Evaluation measures how well retrieval systems rank episodes relative to human relevance judgments.

## Metrics

### Recall@K

**Definition:** % of relevant episodes found in top-K results.

$$\\text{Recall@K} = \\frac{|\\text{relevant in top-K}|}{|\\text{all relevant}|}$$

**Example:**
- Query: "AI in healthcare"
- All relevant episodes: 5
- Top-10 results: 3 are relevant
- Recall@10 = 3/5 = 0.60

**Interpretation:**
- Recall@10 = 0.60 means: "60% of relevant episodes will be found in first 10 results"
- Higher is better; 1.0 = perfect (all relevant found)
- Common thresholds: @1, @5, @10, @50

### Mean Reciprocal Rank (MRR)

**Definition:** Average of 1 / rank of first relevant result.

$$\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank of first relevant in query } i}$$

**Example:**
- Query 1: First relevant at rank 2 → 1/2 = 0.50
- Query 2: First relevant at rank 5 → 1/5 = 0.20
- Query 3: No relevant found → 0
- MRR = (0.50 + 0.20 + 0) / 3 = 0.23

**Interpretation:**
- MRR = 0.80 means: "On average, the first relevant result is at rank 1.25"
- Sensitive to early ranking; values between 0-1
- Useful for "find one good answer quickly" scenarios

### NDCG@K (Normalized Discounted Cumulative Gain)

**Definition:** Ranked quality metric; discounts gains for lower positions.

$$\\text{DCG@K} = \\sum_{i=1}^{K} \\frac{\\text{relevance}(i)}{\\log_2(i+1)}$$

$$\\text{NDCG@K} = \\frac{\\text{DCG@K}}{\\text{DCG@K}_{ideal}}$$

**Example:**
- Relevance grades: {Relevant=1, Somewhat=0.5, NotRelevant=0}
- Top-5 results: [Relevant, NotRelevant, Somewhat, Relevant, NotRelevant]
- DCG@5 = 1/log2(2) + 0/log2(3) + 0.5/log2(4) + 1/log2(5) + 0/log2(6)
         = 1/1 + 0 + 0.5/2 + 1/2.32 + 0
         = 1 + 0.25 + 0.43 = 1.68
- NDCG@5 = DCG@5 / Ideal DCG@5 (where ideal = [Relevant, Relevant, Somewhat, ...])

**Interpretation:**
- NDCG@10 = 0.65 means: "Ranking quality is 65% of ideal"
- Accounts for both finding relevant docs AND ranking them well
- Between 0-1; higher is better

### Coverage

**Definition:** % of episodes with at least one relevant query.

$$\\text{Coverage} = \\frac{|\\text{episodes with any relevant query}|}{|\\text{all episodes}|}$$

**Interpretation:**
- Coverage = 0.15 means: "Only 15% of episodes have a relevant query match"
- Measures if retriever can find episodes when they exist
- Useful for inventory completeness

### Latency (P50, P95, P99)

**Definition:** Response time percentiles across all queries.

**Measurement:**
```python
latencies = [query_time(q) for q in queries]
p50 = np.percentile(latencies, 50)  # Median
p95 = np.percentile(latencies, 95)  # 95% of queries are faster
p99 = np.percentile(latencies, 99)  # 99% of queries are faster
```

**Interpretation:**
- P95 = 150ms means: "95% of queries return in <150ms; 5% are slower"
- Important for user experience; slower = higher bounce rate
- Production target: P95 < 200ms

## Evaluation Harness

### Configuration

```json
{
  "name": "hybrid-rrf-v1",
  "description": "Hybrid RRF retrieval with e5-small embeddings",
  "dataset": "search-bootstrap-v1",
  "relevance": "relevance-pool-v1",
  "methods": [
    {
      "name": "lexical-fts",
      "type": "lexical",
      "config": {
        "language": "english"
      }
    },
    {
      "name": "vector-e5-small",
      "type": "vector",
      "config": {
        "model": "sentence-transformers/e5-small-v2",
        "dimension": 384
      }
    },
    {
      "name": "hybrid-rrf",
      "type": "hybrid",
      "config": {
        "lexical_retriever": "lexical-fts",
        "vector_retriever": "vector-e5-small",
        "rrf_k": 60
      }
    }
  ],
  "metrics": ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "coverage", "latency"]
}
```

### Evaluation Script

```python
# ml/evaluation/run.py
import json
import pandas as pd
from ml.evaluation.metrics import compute_metrics
from ml.retrieval import hybrid_search, lexical_search, vector_search

# Load config
with open("experiments/eval/hybrid-rrf-v1.json") as f:
    config = json.load(f)

# Load data
relevance = pd.read_parquet(f"data/relevance/{config['relevance']}.parquet")
queries = relevance['query'].unique()

# Run evaluation
results = {}
for method_config in config['methods']:
    name = method_config['name']
    method = init_retriever(method_config)
    
    # Retrieve for each query
    rankings = {}
    for query in queries:
        start = time.time()
        ranked_episodes = method.search(query, top_k=50)
        latency = time.time() - start
        
        rankings[query] = {
            'episodes': [ep.id for ep in ranked_episodes],
            'latency': latency
        }
    
    # Compute metrics
    metrics = compute_metrics(
        rankings=rankings,
        relevance=relevance,
        metrics_to_compute=config['metrics']
    )
    
    results[name] = metrics
    
    # Log to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{name}/{metric_name}", metric_value)

# Save results
pd.DataFrame(results).to_csv("experiments/eval/hybrid-rrf-v1-results.csv")
```

### Compute Metrics Function

```python
# ml/evaluation/metrics.py
def compute_metrics(rankings, relevance, metrics_to_compute):
    """Compute metrics across all queries."""
    
    metric_values = {}
    
    for metric in metrics_to_compute:
        if metric.startswith('recall@'):
            k = int(metric.split('@')[1])
            scores = [
                recall_at_k(query, rankings[query]['episodes'], relevance, k)
                for query in rankings
            ]
            metric_values[metric] = np.mean(scores)
        
        elif metric == 'mrr':
            scores = [
                mrr(query, rankings[query]['episodes'], relevance)
                for query in rankings
            ]
            metric_values[metric] = np.mean(scores)
        
        elif metric == 'coverage':
            all_episodes = set()
            for query in rankings:
                all_episodes.update(rankings[query]['episodes'])
            metric_values[metric] = len(all_episodes) / total_episodes
        
        elif metric == 'latency':
            latencies = [rankings[query]['latency'] for query in rankings]
            metric_values[metric] = {
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
            }
    
    return metric_values

def recall_at_k(query, ranked_episodes, relevance_df, k):
    relevant_episodes = relevance_df[
        (relevance_df['query'] == query) &
        (relevance_df['relevance_grade'] == 'relevant')
    ]['episode_id'].unique()
    
    if len(relevant_episodes) == 0:
        return 0.0
    
    top_k_episodes = set(ranked_episodes[:k])
    found = len(top_k_episodes & set(relevant_episodes))
    
    return found / len(relevant_episodes)

def mrr(query, ranked_episodes, relevance_df):
    relevant_episodes = relevance_df[
        (relevance_df['query'] == query) &
        (relevance_df['relevance_grade'] == 'relevant')
    ]['episode_id'].unique()
    
    for rank, episode_id in enumerate(ranked_episodes, 1):
        if episode_id in relevant_episodes:
            return 1.0 / rank
    
    return 0.0
```

### MLflow Integration

Automatically log metrics and hyperparameters:

```python
mlflow.set_experiment("podcast-retrieval")
mlflow.start_run(run_name="hybrid-rrf-v1")

# Log params
mlflow.log_params(config['methods'][2]['config'])

# Log metrics
for method_name, metrics in results.items():
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{method_name}/{metric_name}", value)

# Log artifacts
mlflow.log_artifact("experiments/eval/hybrid-rrf-v1-results.csv")

mlflow.end_run()
```

**Access results:** Open http://localhost:5000 in browser.

## Baseline Results

| Method | Recall@10 | MRR | NDCG@10 | Latency (P95) |
|--------|-----------|-----|---------|---------------|
| Lexical (BM25) | 0.62 | 0.48 | 0.58 | 20ms |
| Vector (e5-small) | 0.58 | 0.42 | 0.52 | 150ms |
| Hybrid (RRF) | 0.70 | 0.55 | 0.64 | 150ms |

(On ~2,500 queries, ~5 relevant episodes per query on average)

## Batch Evaluation

Compare multiple retrieval methods in one run:

```bash
make eval-all  # Runs all configs in experiments/eval/*.json
```

**Output:**
```
Experiment Results:
├── lexical-v1
│   ├── recall@10: 0.62
│   └── mrr: 0.48
├── vector-v1
│   ├── recall@10: 0.58
│   └── mrr: 0.42
└── hybrid-rrf-v1
    ├── recall@10: 0.70
    └── mrr: 0.55
```

## Reproducibility

**Ensure reproducible results:**

1. **Fix random seed:**
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

2. **Version everything:**
   - Dataset version in config
   - Relevance judgment version
   - Model/embedding version
   - Code commit hash (git)

3. **No test set tuning:**
   - Metrics on test set for final reporting only
   - Tuning happens on validation set

4. **Re-run on commit:**
   - CI runs evaluations on each commit
   - Catch metric regressions

## Quality Gates

Before shipping a new retriever:

- [ ] Recall@10 ≥ baseline
- [ ] Latency (P95) < 200ms
- [ ] MRR ≥ prior version
- [ ] Results reproducible (±0.01 across 3 runs)
- [ ] No metric regression on holdout test set

## Future Improvements

1. **Statistical significance testing**: Confidence intervals on metrics
2. **User-side evaluation**: A/B test metrics against actual user feedback
3. **Per-category metrics**: Recall separately for {AI, business, science} categories
4. **User satisfaction**: Implicit signals (click-through, dwell time)
5. **Cost efficiency**: Optimize latency vs. quality tradeoff
