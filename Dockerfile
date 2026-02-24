FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY main.py .
COPY model_registry.py .
COPY requirements.txt .
COPY routers/ routers/

RUN pip install --no-cache-dir -r requirements.txt

# ── Download specialised GGUF models ─────────────────────────────────────────
# Each domain uses the smallest model that handles its task well on 1 vCPU.
# git + github share the same Phi-2 binary to save disk space.
RUN mkdir -p models

# files domain — Qwen2.5-Coder-0.5B Q4_K_M (~340 MB)
RUN wget -q -O models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf"

# git + github domain — Phi-2 Q4_K_M (~1.6 GB)
RUN wget -q -O models/phi-2-q4_k_m.gguf \
    "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"

# rust domain — DeepSeek-Coder-1.3B Q4_K_M (~800 MB)
RUN wget -q -O models/deepseek-coder-1.3b-instruct-q4_k_m.gguf \
    "https://huggingface.co/bartowski/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"

# ── Per-domain environment defaults ──────────────────────────────────────────
ENV FILES_MODEL_PATH=/app/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
ENV FILES_N_CTX=1024
ENV FILES_MAX_TOKENS=128

ENV GIT_MODEL_PATH=/app/models/phi-2-q4_k_m.gguf
ENV GIT_N_CTX=1024
ENV GIT_MAX_TOKENS=128

ENV GITHUB_MODEL_PATH=/app/models/phi-2-q4_k_m.gguf
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
