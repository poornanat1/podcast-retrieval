# Observability: Monitoring, Logging & Tracing

Production-grade observability for PodFind covers metrics, logs, traces, and cost accounting.

## Metrics (Prometheus)

### Retrieval Metrics

```
# Query latency (milliseconds)
retrieval_latency_ms{method=\"lexical\",percentile=\"p50\"} 20
retrieval_latency_ms{method=\"vector\",percentile=\"p95\"} 150
retrieval_latency_ms{method=\"hybrid\",percentile=\"p99\"} 180

# Retrieval errors
retrieval_errors_total{method=\"vector\",error=\"timeout\"} 5
retrieval_errors_total{method=\"lexical\",error=\"invalid_query\"} 2

# Cache hit rate
retrieval_cache_hit_ratio{cache=\"embedding\"} 0.75

# Results returned
retrieval_results_count{method=\"hybrid\",status=\"success\"} 10
retrieval_results_count{method=\"hybrid\",status=\"empty\"} 2
```

### Catalog Metrics

```
# Dataset counts
catalog_shows_total{active=\"true\"} 1287
catalog_episodes_total 542000
catalog_episodes_with_transcripts 48000

# Polling health
catalog_poll_duration_seconds{percentile=\"p95\"} 45
catalog_new_episodes_daily 2500
catalog_poll_errors_total{error=\"timeout\"} 15
catalog_poll_errors_total{error=\"malformed_feed\"} 3

# Deduplication
catalog_dedup_rate 0.08  # 8% duplicates detected

# Freshness
catalog_episodes_hours_since_poll{percentile=\"p95\"} 2
```

### Database Metrics

```
# Connection pool
postgres_pool_size 25
postgres_pool_used 18
postgres_pool_idle 7

# Query performance
postgres_query_duration_seconds{query=\"search_lexical\",percentile=\"p95\"} 0.050
postgres_query_duration_seconds{query=\"search_vector\",percentile=\"p95\"} 0.120
postgres_query_rows{query=\"search_hybrid\"} 10

# Index usage
postgres_index_bloat_ratio 0.05
postgres_table_size_bytes{table=\"episodes\"} 5368709120  # 5GB
```

### ML Metrics

```
# Evaluation metrics (logged by ML pipeline)
eval_recall_at_10{method=\"lexical\"} 0.62
eval_recall_at_10{method=\"vector\"} 0.58
eval_recall_at_10{method=\"hybrid\"} 0.70

eval_mrr{method=\"hybrid\"} 0.55
eval_ndcg_at_10{method=\"hybrid\"} 0.64

eval_coverage{method=\"hybrid\"} 0.42

# Model metrics
embedding_inference_latency_ms{model=\"multilingual-e5-small\",percentile=\"p95\"} 100
embedding_cache_hit_ratio 0.82
```

### Application Metrics

```
# HTTP server
http_request_duration_seconds{method=\"GET\",path=\"/search\",status=\"200\",percentile=\"p95\"} 0.180
http_request_size_bytes{method=\"POST\",path=\"/search\"} 256
http_response_size_bytes{method=\"GET\",path=\"/search\"} 4096

# Worker jobs
worker_job_duration_seconds{worker=\"catalog\",percentile=\"p95\"} 30
worker_job_errors_total{worker=\"embed\"} 2
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'podfind-api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']  # postgres_exporter
    
  - job_name: 'mlflow'
    static_configs:
      - targets: ['localhost:5000']
```

## Logging

### Structured Logging (JSON)

All logs should be structured JSON for easy parsing:

```go
import "github.com/sirupsen/logrus"

log.SetFormatter(&log.JSONFormatter{
    TimestampFormat: time.RFC3339Nano,
})

log.WithFields(log.Fields{
    "query": "AI podcast",
    "method": "hybrid_rrf",
    "latency_ms": 150,
    "results": 10,
    "timestamp": time.Now(),
}).Info("Search query processed")

// Output:
// {"level":"info","query":"AI podcast","method":"hybrid_rrf","latency_ms":150,"results":10,"timestamp":"2026-01-15T10:30:45.123Z","msg":"Search query processed"}
```

### Log Levels

- **ERROR**: Action required; something broke
- **WARN**: Degraded service; recoverable issue
- **INFO**: Normal operation; key events
- **DEBUG**: Detailed diagnostics (disabled in production)

### ELK Stack (Elasticsearch + Logstash + Kibana)

Centralized log aggregation:

```yaml
# docker-compose addition
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  environment:
    discovery.type: single-node
  ports:
    - "9200:9200"

logstash:
  image: docker.elastic.co/logstash/logstash:8.0.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  ports:
    - "5000:5000"

kibana:
  image: docker.elastic.co/kibana/kibana:8.0.0
  ports:
    - "5601:5601"
```

**Logstash pipeline:**
```conf
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  mutate {
    rename => { "message" => "log_message" }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "podfind-%{+YYYY.MM.dd}"
  }
}
```

## Distributed Tracing (Jaeger)

Track requests across services:

```go
import "github.com/uber/jaeger-client-go"

// Initialize tracer
closer, err := jaeger.InitGlobal(
    "podfind-api",
    jaeger.InitGlobalTracer(),
)
defer closer.Close()

// Trace a search request
span, ctx := opentracing.StartSpanFromContext(ctx, "search")
defer span.Finish()

span.SetTag("query", query)
span.SetTag("method", "hybrid")

// Nested spans
childSpan, ctx := opentracing.StartSpanFromContext(ctx, "lexical_search")
results := lexicalSearch(ctx, query)
childSpan.Finish()
```

**Jaeger UI:** http://localhost:16686

## Dashboards (Grafana)

### Dashboard: Search Performance

```json
{
  "title": "Search Performance",
  "panels": [
    {
      "title": "Query Latency (P95)",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, retrieval_latency_ms)"
        }
      ]
    },
    {
      "title": "Retrieval Errors",
      "targets": [
        {
          "expr": "rate(retrieval_errors_total[5m])"
        }
      ]
    },
    {
      "title": "Cache Hit Rate",
      "targets": [
        {
          "expr": "retrieval_cache_hit_ratio"
        }
      ]
    }
  ]
}
```

### Dashboard: Catalog Health

```json
{
  "title": "Catalog Health",
  "panels": [
    {
      "title": "Shows & Episodes",
      "targets": [
        {"expr": "catalog_shows_total"},
        {"expr": "catalog_episodes_total"}
      ]
    },
    {
      "title": "Poll Errors (24h)",
      "targets": [
        {"expr": "increase(catalog_poll_errors_total[24h])"}
      ]
    },
    {
      "title": "Transcript Coverage",
      "targets": [
        {"expr": "catalog_episodes_with_transcripts / catalog_episodes_total"}
      ]
    }
  ]
}
```

## Alerting

### Alert Rules (Prometheus)

```yaml
# alerts.yml
groups:
  - name: podfind
    rules:
      - alert: HighQueryLatency
        expr: histogram_quantile(0.95, retrieval_latency_ms) > 500
        for: 5m
        annotations:
          summary: "Query latency high ({{ $value }}ms)"
          
      - alert: CatalogPollFailing
        expr: increase(catalog_poll_errors_total[1h]) > 100
        for: 10m
        annotations:
          summary: "Catalog polling experiencing errors"
          
      - alert: DatabaseConnectionLeakage
        expr: postgres_pool_used > postgres_pool_size * 0.9
        for: 5m
        annotations:
          summary: "Database connection pool near capacity"
          
      - alert: EmbeddingCacheHitLow
        expr: embedding_cache_hit_ratio < 0.7
        for: 30m
        annotations:
          summary: "Embedding cache hit rate below threshold"
```

### Notification Channels

- **Slack**: #alerts channel
- **PagerDuty**: On-call rotation for critical alerts
- **Email**: Daily digest of warnings

**Grafana alert notification:**
```yaml
notificationChannels:
  - name: slack-alerts
    type: slack
    settings:
      url: https://hooks.slack.com/services/...
```

## Cost Tracking

### ML Pipeline Costs

Track embedding and model inference costs:

```python
# ml/observability/cost.py
import mlflow

def log_embedding_cost(num_tokens: int, model: str):
    """Log embedding API cost."""
    # multilingual-e5-small: free (local inference)
    # Claude API: $0.075 per 1M tokens
    if "claude" in model:
        cost = num_tokens * 0.075 / 1e6
        mlflow.log_metric("embedding_cost_usd", cost)

def log_training_cost(duration_hours: float, gpu_hours: float):
    """Log training infrastructure cost."""
    # GPU cost: $1/hour (NVIDIA A100)
    cost = gpu_hours * 1.0
    mlflow.log_metric("training_cost_usd", cost)
```

**Track in MLflow:**
- Cost per experiment run
- Cost per dataset build
- Cost per evaluation pass

### Infrastructure Costs

```
Database (Postgres):        $50/month
Object Storage (MinIO):     $20/month
Kubernetes cluster:         $200/month
Monitoring/Logging:         $100/month
CDN (if used):             $50/month
─────────────────────────────────
Total:                      ~$420/month
```

## SLOs & Error Budgets

### Service Level Objectives

| Metric | SLO | Error Budget |
|--------|-----|--------------|
| Availability | 99.5% | 3.6 hours/month |
| P95 Latency | < 200ms | 5% of queries |
| Error Rate | < 0.1% | 0.1% of queries |

### Tracking SLO Compliance

```python
import datetime

compliance_rate = successful_requests / total_requests
if compliance_rate >= 0.995:
    print("✓ Availability SLO met")
else:
    error_budget_used = (1 - compliance_rate) * 100
    print(f"✗ Error budget at {error_budget_used}% for {datetime.date.today().month}")
```

## Health Checks

### Readiness Probe

Service is ready to serve traffic:

```go
func readinessCheck(w http.ResponseWriter, r *http.Request) {
    checks := map[string]bool{
        "database": db.Ping(r.Context()) == nil,
        "cache":    cache.Ping(r.Context()) == nil,
        "embedding_model": modelLoaded,
    }
    
    allHealthy := true
    for _, healthy := range checks {
        if !healthy {
            allHealthy = false
            break
        }
    }
    
    if allHealthy {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    json.NewEncoder(w).Encode(checks)
}
```

### Liveness Probe

Service is still running:

```go
func livenessCheck(w http.ResponseWriter, r *http.Request) {
    // Simple check: can we acquire a database connection?
    if err := db.Ping(r.Context()); err != nil {
        w.WriteHeader(http.StatusServiceUnavailable)
        return
    }
    
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "alive"})
}
```

## OnCall & Incident Response

### Incident Severity

- **P1 (Critical)**: Service down or severely degraded (SLA violated)
- **P2 (High)**: Major feature broken; users impacted
- **P3 (Medium)**: Minor issue; workaround available
- **P4 (Low)**: Documentation, nice-to-have fix

### Incident Response Playbook

1. **Detect**: Alert fires in Prometheus
2. **Page**: PagerDuty notifies on-call engineer
3. **Assess**: Check dashboards, logs, traces
4. **Respond**:
   - Rollback if recent deployment caused issue
   - Scale up if capacity issue
   - Query kill if slow query blocking
5. **Resolve**: Fix underlying cause
6. **Postmortem**: Prevent recurrence

### Useful Queries

**Check database connection pool:**
```sql
SELECT datname, count(*) as connections
FROM pg_stat_activity
GROUP BY datname;
```

**Find slow queries:**
```sql
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Monitor replication lag:**
```sql
SELECT slot_name, restart_lsn, confirmed_flush_lsn
FROM pg_replication_slots;
```

## Future Improvements

1. **Continuous Profiling**: Track CPU/memory usage patterns
2. **Synthetic Monitoring**: Simulate user queries; track availability
3. **Error Context**: Capture request body + stack traces on failures
4. **Customer Impact**: Segment metrics by user/region
5. **Cost Optimization**: Alert when costs exceed forecast
