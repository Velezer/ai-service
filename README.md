# ai-service

FastAPI service backed by `llama-cpp-python` for local code generation.

## 1 vCPU Rust-focused fast mode

This project is tuned for low-latency code generation on small CPU machines:

- Uses `Qwen2.5-Coder-0.5B-Instruct-Q4_K_M` by default.
- Uses smaller context (`N_CTX=1024`) and token budgets (`MAX_TOKENS=128`).
- Adds a dedicated endpoint for Rust-only fast generation.

### Rust fast endpoint

`POST /generate/code/rust/fast`

Request body:

```json
{
  "instruction": "Write a function that parses CSV lines into a struct",
  "context": "No external crates"
}
```

Response:

```json
{
  "response": "...rust code...",
  "language": "rust",
  "mode": "fast"
}
```

### General code endpoint with mode

`POST /generate/code`

Use:

- `mode: "quality"` for normal output.
- `mode: "fast"` for smaller, quicker output.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
