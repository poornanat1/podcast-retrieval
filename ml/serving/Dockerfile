# Model server: ONNX Runtime inference for the promoted bundle.
# Build from the repository root:
#   docker build -f ml/serving/Dockerfile -t podfind-model-server .
# The bundle is fetched from the object store at start (PODFIND_BUNDLE_ID)
# into /data/bundles, a volume, so the image holds no model weights.
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 PYTHONPATH=/app PATH=/app/.venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --only-group serving
COPY ml ./ml
COPY deploy/railway/model-server-entrypoint.sh /entrypoint.sh
# Runs as root: Railway mounts volumes owned by root, and /data is one.
RUN mkdir -p /data/bundles
ENV PODFIND_BUNDLE_DIR=/data/bundles PORT=8081
EXPOSE 8081
ENTRYPOINT ["/entrypoint.sh"]
