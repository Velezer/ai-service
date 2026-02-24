FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies in an isolated path so the runtime image
# doesn't need compiler toolchains.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python deps from the builder stage.
COPY --from=builder /install /usr/local

# Copy application code
COPY main.py .
COPY model_registry.py .
COPY requirements.txt .
COPY routers/ routers/

# ── Download specialised GGUF models ─────────────────────────────────────────
# Each domain uses a compact model profile to stay within image size limits.
# files + git + github share one Qwen binary to avoid duplicate large artifacts.
RUN mkdir -p models

# rust domain — DeepSeek-Coder-1.3B Q4_K_M (~800 MB)
RUN wget -q -O models/deepseek-coder-1.3b-instruct-q4_k_m.gguf \
    "https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"

# files domain — Qwen2.5-Coder-0.5B Q4_K_M (~340 MB)
RUN wget -q -O models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf"


# ── Per-domain environment defaults ──────────────────────────────────────────
ENV FILES_MODEL_PATH=/app/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
ENV FILES_N_CTX=1024
ENV FILES_MAX_TOKENS=128

ENV GIT_MODEL_PATH=/app/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
ENV GIT_N_CTX=1024
ENV GIT_MAX_TOKENS=128

ENV GITHUB_MODEL_PATH=/app/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
ENV GITHUB_N_CTX=1024
ENV GITHUB_MAX_TOKENS=128

ENV RUST_MODEL_PATH=/app/models/deepseek-coder-1.3b-instruct-q4_k_m.gguf
ENV RUST_N_CTX=2048
ENV RUST_MAX_TOKENS=256

# Shared inference settings optimised for 1 vCPU
ENV N_THREADS=1
ENV N_BATCH=128
ENV USE_MLOCK=true
# Set EXCLUSIVE_LOADING=true if RAM < 4 GB to unload unused models automatically
ENV EXCLUSIVE_LOADING=false

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
