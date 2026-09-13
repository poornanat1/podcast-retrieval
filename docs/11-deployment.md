# Deployment: Local, Docker & Kubernetes

PodFind can be deployed locally (development), via Docker Compose, or to Kubernetes.

## Local Development

### Prerequisites

- Go 1.26+
- Python 3.11+
- PostgreSQL 15+ with pgvector
- uv (Python package manager)

### Setup

```bash
# Clone repository
git clone https://github.com/your/podcast-retrieval.git
cd podcast-retrieval

# Create .env file
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Install dependencies
uv sync
go mod download

# Run migrations
make migrate

# Start background workers (in separate terminals)
go run ./cmd/catalog-worker   # RSS polling
go run ./cmd/embed-worker     # Embedding inference
go run ./cmd/serve            # API server

# In Python: Run evaluation
make eval-all
```

## Docker Compose (Recommended for Local/Staging)

**Services:**
- PostgreSQL 15 (with pgvector)
- MinIO (S3-compatible object storage)
- MLflow (experiment tracking)
- PodFind API (Go)
- PodFind Workers (Go)

### Start Stack

```bash
docker compose up -d
```

**Access:**
- API: http://localhost:8080
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001 (credentials: minioadmin/minioadmin)

### docker-compose.yml Structure

```yaml
version: '3.8'
services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_DB: podfind
      POSTGRES_USER: podfind
      POSTGRES_PASSWORD: podfind
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  mlflow:
    image: python:3.11
    command: >
      bash -c 'pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000'
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/data

  api:
    build: .
    command: go run ./cmd/serve
    environment:
      DATABASE_URL: postgres://podfind:podfind@postgres:5432/podfind?sslmode=disable
      PORT: 8080
    ports:
      - "8080:8080"
    depends_on:
      - postgres
      - minio

volumes:
  postgres_data:
  minio_data:
  mlflow_data:
```

### Shut Down

```bash
docker compose down
# Keep volumes: docker compose down -v  # Remove data
```

## Production: Kubernetes

### Helm Chart Structure

```
deploy/helm/podfind/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── templates/
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── postgres-statefulset.yaml
│   ├── api-deployment.yaml
│   ├── worker-deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── hpa.yaml
```

### Deployment

```bash
# Install chart
helm install podfind ./deploy/helm/podfind \
  -f deploy/helm/podfind/values-prod.yaml \
  --namespace podfind \
  --create-namespace

# Upgrade
helm upgrade podfind ./deploy/helm/podfind \
  -f deploy/helm/podfind/values-prod.yaml \
  --namespace podfind
```

### values-prod.yaml

```yaml
replicaCount: 3

image:
  repository: your-registry/podfind
  tag: latest
  pullPolicy: IfNotPresent

postgresql:
  enabled: true
  auth:
    password: ${PG_PASSWORD}  # From secret
  primary:
    persistence:
      size: 100Gi

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: podfind.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Health Checks

```go
// cmd/serve/main.go
func healthcheck(w http.ResponseWriter, r *http.Request) {
    // Check database connection
    if err := db.Ping(r.Context()); err != nil {
        w.WriteHeader(http.StatusServiceUnavailable)
        json.NewEncoder(w).Encode(map[string]string{"status": "unhealthy"})
        return
    }
    
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

router.HandleFunc("/health", healthcheck)
```

**Kubernetes probe:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
```

## Building & Publishing Container Images

### Dockerfile

```dockerfile
# Build stage
FROM golang:1.26 as builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o api ./cmd/serve

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/api /usr/local/bin/
EXPOSE 8080
CMD ["api"]
```

### Build & Push

```bash
# Build
docker build -t your-registry/podfind:latest .

# Push
docker push your-registry/podfind:latest

# Or via GitHub Actions
# .github/workflows/docker.yml (auto-builds on push)
```

## Environment Configuration

### .env.example

```bash
# Database
DATABASE_URL=postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable

# Object storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
PORT=8080
LOG_LEVEL=info

# Features
ENABLE_VOICE_SEARCH=false
ENABLE_RERANKER=false
```

### Secrets Management (Production)

```bash
# Create Kubernetes secret
kubectl create secret generic podfind-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --from-literal=minio-access-key=$MINIO_ACCESS_KEY \
  --namespace podfind

# Reference in Helm template
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: podfind-secrets
        key: database-url
```

## Upgrades & Rollbacks

### Rolling Deployment

Kubernetes automatically performs rolling updates:
```bash
kubectl set image deployment/podfind \
  podfind=your-registry/podfind:v2 \
  --namespace podfind
```

**Progress:** Watch rollout
```bash
kubectl rollout status deployment/podfind --namespace podfind
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/podfind --namespace podfind

# Rollback to specific revision
kubectl rollout history deployment/podfind
kubectl rollout undo deployment/podfind --to-revision=5
```

## Monitoring & Observability

### Prometheus Metrics

Expose Prometheus metrics on `/metrics`:

```go
import "github.com/prometheus/client_golang/prometheus/promhttp"

router.Handle("/metrics", promhttp.Handler())
```

**Configure scraping:**
```yaml
scrape_configs:
  - job_name: podfind
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Logging

Send structured logs to centralized system:

```go
import "github.com/sirupsen/logrus"

log.SetFormatter(&log.JSONFormatter{})

log.WithFields(log.Fields{
    "query": "AI podcast",
    "latency_ms": 150,
    "method": "hybrid_rrf",
}).Info("Search query processed")
```

**ELK Stack or Datadog integration for collection.**

## Database Backup & Restore

### Automated Backups (Production)

```bash
# Daily backup to cloud storage
0 2 * * * pg_dump podfind | gzip | aws s3 cp - s3://backups/podfind-$(date +\%Y\%m\%d).sql.gz
```

### Restore from Backup

```bash
# Download backup
aws s3 cp s3://backups/podfind-20260115.sql.gz .

# Restore
gunzip podfind-20260115.sql.gz
psql podfind < podfind-20260115.sql
```

## Cost Optimization

### Development
- Single machine: 2 vCPU, 4GB RAM (~\$20/month)
- Database: Shared PostgreSQL (~\$10/month)
- Storage: 10GB (~\$1/month)
- **Total:** ~\$30/month

### Staging
- 2 API nodes: 2 vCPU, 2GB RAM each (~\$40/month)
- Database: Managed PostgreSQL (~\$50/month)
- Load balancer (~\$20/month)
- **Total:** ~\$110/month

### Production
- 5 API nodes: 2 vCPU, 4GB RAM each (~\$200/month)
- Database: HA PostgreSQL with replication (~\$150/month)
- Load balancer (~\$20/month)
- CDN/Object storage (~\$50/month)
- Monitoring/logging (~\$100/month)
- **Total:** ~\$520/month

## Disaster Recovery

### RTO/RPO Targets
- **RTO (Recovery Time Objective):** < 1 hour
- **RPO (Recovery Point Objective):** < 15 minutes

### Strategy
1. Automated daily backups to S3
2. Hourly incremental backups
3. Multi-region replication (for critical data)
4. Documented restore procedure (tested monthly)

## Future: Multi-Region Deployment

```yaml
# Deploy same stack to multiple regions
regions:
  - us-east-1
  - eu-west-1
  - ap-southeast-1

# Global load balancer routes to closest region
# Replication syncs data across regions
```

See [12-observability.md](12-observability.md) for monitoring production deployments.
