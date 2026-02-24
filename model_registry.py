"""
model_registry.py
-----------------
Lazy-loading model registry for 1-vCPU deployments.

Each domain (files, git, github, rust) maps to its own GGUF model.
Models are loaded on first request and cached for subsequent calls.
On a memory-constrained host you can set EXCLUSIVE_LOADING=true to
unload the previous model before loading a new one.
"""

import os
import threading
from typing import Optional
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

# ── shared inference defaults ────────────────────────────────────────────────
N_THREADS      = int(os.getenv("N_THREADS", 1))
N_BATCH        = int(os.getenv("N_BATCH", 128))
USE_MLOCK      = os.getenv("USE_MLOCK", "true").lower() in {"1", "true", "yes"}
USE_MMAP       = os.getenv("USE_MMAP", "true").lower() in {"1", "true", "yes"}
# When True, unload the current model before loading a different one.
# Saves RAM at the cost of a cold-start delay on domain switches.
EXCLUSIVE_LOADING = os.getenv("EXCLUSIVE_LOADING", "false").lower() in {"1", "true", "yes"}

# ── per-domain model paths & context sizes ───────────────────────────────────
DOMAIN_CONFIG: dict[str, dict] = {
    "files": {
        "model_path": os.getenv(
            "FILES_MODEL_PATH",
            "./models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        ),
        "n_ctx": int(os.getenv("FILES_N_CTX", 128)),
        "max_tokens": int(os.getenv("FILES_MAX_TOKENS", 128)),
        "temperature": float(os.getenv("FILES_TEMPERATURE", 0.1)),
        "top_p": float(os.getenv("FILES_TOP_P", 0.5)),
    },
    "git": {
        "model_path": os.getenv(
            "GIT_MODEL_PATH",
            "./models/phi-2-q4_k_m.gguf",
        ),
        "n_ctx": int(os.getenv("GIT_N_CTX", 1024)),
        "max_tokens": int(os.getenv("GIT_MAX_TOKENS", 128)),
        "temperature": float(os.getenv("GIT_TEMPERATURE", 0.2)),
        "top_p": float(os.getenv("GIT_TOP_P", 0.95)),
    },
    "github": {
        # Reuses the same model file as git by default (shared binary)
        "model_path": os.getenv(
            "GITHUB_MODEL_PATH",
            "./models/phi-2-q4_k_m.gguf",
        ),
        "n_ctx": int(os.getenv("GITHUB_N_CTX", 1024)),
        "max_tokens": int(os.getenv("GITHUB_MAX_TOKENS", 128)),
        "temperature": float(os.getenv("GITHUB_TEMPERATURE", 0.3)),
        "top_p": float(os.getenv("GITHUB_TOP_P", 0.95)),
    },
    "rust": {
        "model_path": os.getenv(
            "RUST_MODEL_PATH",
            "./models/deepseek-coder-1.3b-instruct-q4_k_m.gguf",
        ),
        "n_ctx": int(os.getenv("RUST_N_CTX", 2048)),
        "max_tokens": int(os.getenv("RUST_MAX_TOKENS", 256)),
        "temperature": float(os.getenv("RUST_TEMPERATURE", 0.1)),
        "top_p": float(os.getenv("RUST_TOP_P", 0.90)),
    },
}

# ── internal state ────────────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_loaded: dict[str, Optional[Llama]] = {k: None for k in DOMAIN_CONFIG}
_active_domain: Optional[str] = None  # tracked only when EXCLUSIVE_LOADING=True


def get_model(domain: str) -> Llama:
    """Return the Llama instance for *domain*, loading it lazily if needed."""
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {list(DOMAIN_CONFIG)}")

    with _lock:
        if _loaded[domain] is not None:
            return _loaded[domain]  # type: ignore[return-value]

        cfg = DOMAIN_CONFIG[domain]

        if EXCLUSIVE_LOADING:
            _unload_all_except(domain)

        _loaded[domain] = Llama(
            model_path=cfg["model_path"],
            n_ctx=cfg["n_ctx"],
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            use_mlock=USE_MLOCK,
            verbose=False,
            mmap=USE_MMAP,
        )
        return _loaded[domain]  # type: ignore[return-value]


def get_cfg(domain: str) -> dict:
    """Return the config dict for *domain* (does NOT load the model)."""
    return DOMAIN_CONFIG[domain]


def _unload_all_except(keep: str) -> None:
    """Free every loaded model except *keep*. Call only while holding _lock."""
    global _active_domain
    for d, instance in _loaded.items():
        if d != keep and instance is not None:
            del instance
            _loaded[d] = None
    _active_domain = keep
