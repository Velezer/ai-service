"""
main.py
-------
Entry point for the multi-domain LLM API.

Each domain (files, git, github, rust) uses its own lazily-loaded GGUF model
so only the models that are actually called consume memory.  On a 1-vCPU host
this keeps cold-start time low and avoids loading unused weights.

Cloud AI routers connect to external APIs (all have free tiers):
  /chatgpt/*  – OpenAI ChatGPT   (gpt-4o-mini)
  /gemini/*   – Google Gemini    (gemini-1.5-flash)
  /groq/*     – Groq Cloud       (llama-3.1-8b-instant, ultra-fast)
  /claude/*   – Anthropic Claude (claude-3-haiku)

Local GGUF routers (lazy-loaded, zero RAM when unused):
  /files/*    – basic file operations   (Qwen2.5-Coder-0.5B Q4)
  /git/*      – git helpers             (Phi-2 Q4)
  /github/*   – GitHub helpers          (Phi-2 Q4, shared with git)
  /rust/*     – Rust code generation    (DeepSeek-Coder-1.3B Q4)
"""

from fastapi import FastAPI
from model_registry import DOMAIN_CONFIG, _loaded
from routers import files, git, github, rust
from routers import openai_chat, gemini, groq_chat, claude

app = FastAPI(
    title="Multi-Domain LLM API",
    description=(
        "Local GGUF endpoints (files, git, GitHub, Rust) plus cloud AI integrations "
        "with free tiers: OpenAI ChatGPT, Google Gemini, Groq, and Anthropic Claude."
    ),
    version="2.2.0",
)

# ── local GGUF domain routers ─────────────────────────────────────────────────
app.include_router(files.router)
app.include_router(git.router)
app.include_router(github.router)
app.include_router(rust.router)

# ── cloud AI routers (free-tier APIs) ────────────────────────────────────────
app.include_router(openai_chat.router)
app.include_router(gemini.router)
app.include_router(groq_chat.router)
app.include_router(claude.router)


# ── health / status ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/status")
def status():
    """Show which domain models are currently loaded in memory."""
    return {
        "loaded_models": {
            domain: (instance is not None)
            for domain, instance in _loaded.items()
        },
        "model_paths": {
            domain: cfg["model_path"]
            for domain, cfg in DOMAIN_CONFIG.items()
        },
    }
