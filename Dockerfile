FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y git build-essential curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy FastAPI code and requirements
COPY main.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and compile llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /app/llama.cpp
RUN cd /app/llama.cpp && make

# Download tiny GGUF model during build
RUN mkdir -p models && \
    wget -O models/ggml-pythia-70m-deduped-q4_0.bin https://huggingface.co/nomic-ai/gpt4all-j/resolve/main/ggml/ggml-pythia-70m-deduped-q4_0.bin

# Expose port
EXPOSE 8000

# Use PORT from environment (Railway sets it)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]