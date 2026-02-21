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


# Download tiny model
RUN mkdir -p models && \
    wget -O models/ggml-pythia-70m-deduped-q4_0.bin \
    https://huggingface.co/mradermacher/pythia-70m-deduped-v0-GGUF/resolve/main/pythia-70m-deduped-v0.f16.gguf
    # https://huggingface.co/Crataco/Pythia-Deduped-Series-GGML/blob/main/ggmlv3-pythia-70m-deduped-q5_1.bin

    
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]