# Speech server: Whisper on CPU (or the OpenAI backend) for voice search.
# Build from the repository root:
#   docker build -f ml/speech/Dockerfile -t podfind-speech-server .
# Model weights download from the Hugging Face hub on first start into
# HF_HOME (/data/hf, a volume), so later starts are offline and fast.
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 PYTHONPATH=/app PATH=/app/.venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --only-group speech
COPY ml ./ml
# Runs as root: Railway mounts volumes owned by root, and /data is one.
RUN mkdir -p /data/hf
ENV HF_HOME=/data/hf PODFIND_STT_DEVICE=cpu OMP_NUM_THREADS=4 PORT=8082
EXPOSE 8082
CMD ["sh", "-c", "exec python -m ml.speech.server --host :: --port ${PORT:-8082}"]
