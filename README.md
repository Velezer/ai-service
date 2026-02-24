# ai-service

FastAPI service backed by `llama-cpp-python` for local AI assistance, optimised for **1 vCPU** deployments.

## Architecture — separated domain models

Instead of one large model doing everything, each functional domain loads its own small, specialised GGUF model **lazily on first request**.  Unused domains consume zero RAM.

| Domain | Model | Size (Q4_K_M) | Strengths |
|--------|-------|--------------|-----------|
| `files` | Qwen2.5-Coder-0.5B | ~340 MB | File ops, path manipulation, filtering |
| `git` | Phi-2 | ~1.6 GB | Commit messages, diffs, branch names |
| `github` | Phi-2 (shared) | — | PR descriptions, issue titles, release notes |
| `rust` | DeepSeek-Coder-1.3B | ~800 MB | Rust generation, fixing, refactoring, tests |

> **Memory tip:** Set `EXCLUSIVE_LOADING=true` in `.env` to automatically unload the previous domain model before loading a new one.  Saves ~300–600 MB at the cost of a cold-start delay on domain switches.

---

## Endpoints

### `/files` — Basic file operations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/files/generate` | General file-operation task |
| POST | `/files/rename` | Suggest new filenames from a rule |
| POST | `/files/filter` | Filter a filename list by criteria |
| POST | `/files/stream` | Streaming file-operation generation |

**Example — rename files:**
```json
POST /files/rename
{
  "files": ["user_data.CSV", "order_log.CSV"],
  "rule": "lowercase extension and add date prefix 2024-01-"
}
```

---

### `/git` — Git helpers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/git/commit-message` | Generate a commit message from a diff |
| POST | `/git/diff-summary` | Summarise a diff as bullet points |
| POST | `/git/branch-name` | Suggest a branch name |
| POST | `/git/command` | Suggest git command(s) for a task |
| POST | `/git/stream/commit-message` | Streaming commit message |

**Example — commit message:**
```json
POST /git/commit-message
{
  "diff": "diff --git a/src/main.rs ...",
  "style": "conventional"
}
```

---

### `/github` — GitHub helpers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/github/pr-description` | Generate a PR description |
| POST | `/github/issue-title` | Generate an issue title |
| POST | `/github/release-notes` | Generate release notes from commits |
| POST | `/github/review-comment` | Write a PR review comment |
| POST | `/github/label-suggestion` | Suggest labels for an issue/PR |
| POST | `/github/stream/pr-description` | Streaming PR description |

**Example — PR description:**
```json
POST /github/pr-description
{
  "title": "Add CSV export",
  "diff_summary": "Added export_csv() to DataTable, wired to /export route",
  "issue_ref": "#42",
  "style": "concise"
}
```

---

### `/rust` — Rust code generation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/rust/generate` | Generate Rust code |
| POST | `/rust/fix` | Fix compiler errors |
| POST | `/rust/doc` | Add doc-comments |
| POST | `/rust/refactor` | Refactor toward a goal |
| POST | `/rust/test` | Generate unit tests |
| POST | `/rust/stream/generate` | Streaming Rust generation |

**Example — generate:**
```json
POST /rust/generate
{
  "instruction": "Write a function that parses CSV lines into a Vec<Record>",
  "context": "No external crates, use std only",
  "mode": "fast"
}
```

---

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/status` | Shows which domain models are loaded |

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key settings:

```env
N_THREADS=1          # match your vCPU count
N_BATCH=128          # lower = less RAM pressure
EXCLUSIVE_LOADING=false  # set true if RAM < 4 GB
```

Each domain has its own `*_MODEL_PATH`, `*_N_CTX`, `*_MAX_TOKENS`, `*_TEMPERATURE`, `*_TOP_P` variables.

---

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t ai-service .
docker run -p 8000:8000 ai-service
```

Interactive docs: http://localhost:8000/docs
