FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python API
COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download a stronger code-capable model
RUN mkdir -p models && \
    wget -O models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf

ENV MODEL_PATH=/app/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
