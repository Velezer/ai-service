FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python API
COPY main.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download prebuilt llama.cpp release (Linux x86_64)
RUN mkdir -p llama && \
    wget -O llama.tar.gz \
    https://github.com/ggml-org/llama.cpp/releases/download/b8121/llama-b8121-bin-ubuntu-x64.tar.gz && \
    tar -xzf llama.tar.gz -C llama && \
    rm llama.tar.gz && \
    chmod +x llama/llama-cli

# Download tiny model
RUN mkdir -p models && \
    wget -O models/ggml-pythia-70m-deduped-q4_0.bin \
    https://huggingface.co/nomic-ai/gpt4all-j/resolve/main/ggml/ggml-pythia-70m-deduped-q4_0.bin

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]