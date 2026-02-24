"""
main.py
-------
Entry point for the multi-domain LLM API.

Each domain (files, git, github, rust) uses its own lazily-loaded GGUF model
so only the models that are actually called consume memory.  On a 1-vCPU host
this keeps cold-start time low and avoids loading unused weights.

Routers
-------
  /files/*   – basic file operations   (Qwen2.5-Coder-0.5B Q4)
  /git/*     – git helpers             (Phi-2 Q4)
  /github/*  – GitHub helpers          (Phi-2 Q4, shared with git)
  /rust/*    – Rust code generation    (DeepSeek-Coder-1.3B Q4)
"""

from fastapi import FastAPI
from model_registry import DOMAIN_CONFIG, _loaded
from routers import files, git, github, rust

app = FastAPI(
    title="Multi-Domain LLM API",
    description=(
        "Specialised LLM endpoints for file operations, git, GitHub, and Rust. "
        "Each domain uses a dedicated small model loaded lazily on first request."
    ),
    version="2.0.0",
)

# ── mount domain routers ──────────────────────────────────────────────────────
app.include_router(files.router)
app.include_router(git.router)
app.include_router(github.router)
app.include_router(rust.router)


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
