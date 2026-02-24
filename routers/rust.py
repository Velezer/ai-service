"""
routers/rust.py
---------------
Rust-specific code generation endpoints.
Model: DeepSeek-Coder-1.3B (Q4_K_M) — best small model for Rust on 1 vCPU.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from model_registry import get_model, get_cfg

router = APIRouter(prefix="/rust", tags=["rust"])
DOMAIN = "rust"

_STOP_TOKENS = ["\n\n```", "\n\nExplanation:", "\n\nNotes:", "\n\n//"]


# ── request schemas ───────────────────────────────────────────────────────────

class RustGenerateRequest(BaseModel):
    instruction: str = Field(..., description="Describe the Rust code to generate")
    context: Optional[str] = Field(default=None, description="Crate constraints, existing code, etc.")
    mode: Literal["quality", "fast"] = Field(default="quality")


class RustFixRequest(BaseModel):
    code: str = Field(..., description="Rust code with errors")
    error: str = Field(..., description="Compiler error message (rustc / cargo output)")


class RustDocRequest(BaseModel):
    code: str = Field(..., description="Rust function or struct to document")
    style: Literal["doc-comment", "inline"] = Field(default="doc-comment")


class RustRefactorRequest(BaseModel):
    code: str = Field(..., description="Rust code to refactor")
    goal: str = Field(..., description="Refactoring goal e.g. 'use iterators', 'reduce allocations'")


class RustTestRequest(BaseModel):
    code: str = Field(..., description="Rust function to write tests for")
    framework: Literal["std", "tokio"] = Field(default="std")


# ── prompt builders ───────────────────────────────────────────────────────────

def _build_generate_prompt(req: RustGenerateRequest) -> str:
    ctx = f"\nConstraints:\n{req.context.strip()}\n" if req.context else ""
    verbosity = (
        "Return only valid Rust code. No explanations. Prefer std library."
        if req.mode == "fast"
        else "Return production-quality Rust code with brief inline comments where helpful."
    )
    return (
        f"{verbosity}\n"
        f"Task: {req.instruction.strip()}"
        f"{ctx}\n"
        "```rust"
    )


def _build_fix_prompt(req: RustFixRequest) -> str:
    return (
        "Fix the following Rust code. Output only the corrected code — no explanations.\n\n"
        f"Error:\n{req.error.strip()}\n\n"
        f"Code:\n```rust\n{req.code.strip()}\n```\n\n"
        "Fixed code:\n```rust"
    )


def _build_doc_prompt(req: RustDocRequest) -> str:
    style_hint = (
        "Add /// doc-comments above each public item."
        if req.style == "doc-comment"
        else "Add // inline comments explaining non-obvious lines."
    )
    return (
        f"Add Rust documentation comments to the following code. {style_hint}\n"
        "Output only the documented code.\n\n"
        f"```rust\n{req.code.strip()}\n```\n\n"
        "Documented code:\n```rust"
    )


def _build_refactor_prompt(req: RustRefactorRequest) -> str:
    return (
        f"Refactor the following Rust code. Goal: {req.goal.strip()}\n"
        "Output only the refactored code — no explanations.\n\n"
        f"```rust\n{req.code.strip()}\n```\n\n"
        "Refactored code:\n```rust"
    )


def _build_test_prompt(req: RustTestRequest) -> str:
    async_hint = (
        "#[tokio::test]\nasync fn" if req.framework == "tokio" else "#[test]\nfn"
    )
    return (
        "Write Rust unit tests for the following function.\n"
        f"Use {req.framework} test framework. Test happy path and edge cases.\n"
        "Output only the test module.\n\n"
        f"```rust\n{req.code.strip()}\n```\n\n"
        f"```rust\n#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    {async_hint}"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokens(mode: str) -> tuple[int, float]:
    cfg = get_cfg(DOMAIN)
    if mode == "fast":
        return max(96, cfg["max_tokens"] // 2), min(cfg["temperature"], 0.1)
    return cfg["max_tokens"], cfg["temperature"]


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate")
def rust_generate(req: RustGenerateRequest):
    """Generate Rust code from an instruction."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    max_tokens, temperature = _tokens(req.mode)
    output = llm(
        _build_generate_prompt(req),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=cfg["top_p"],
        stop=_STOP_TOKENS,
        echo=False,
    )
    return {
        "response": output["choices"][0]["text"].strip(),
        "language": "rust",
        "mode": req.mode,
    }


@router.post("/fix")
def rust_fix(req: RustFixRequest):
    """Fix Rust compiler errors."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_fix_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=min(cfg["temperature"], 0.1),
        top_p=cfg["top_p"],
        stop=_STOP_TOKENS,
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "language": "rust"}


@router.post("/doc")
def rust_doc(req: RustDocRequest):
    """Add documentation comments to Rust code."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_doc_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=_STOP_TOKENS,
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "style": req.style}


@router.post("/refactor")
def rust_refactor(req: RustRefactorRequest):
    """Refactor Rust code toward a stated goal."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_refactor_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=_STOP_TOKENS,
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "language": "rust"}


@router.post("/test")
def rust_test(req: RustTestRequest):
    """Generate unit tests for a Rust function."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_test_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n}\n}"],
        echo=False,
    )
    return {
        "response": output["choices"][0]["text"].strip(),
        "language": "rust",
        "framework": req.framework,
    }


@router.post("/stream/generate")
def rust_stream_generate(req: RustGenerateRequest):
    """Streaming Rust code generation."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    max_tokens, temperature = _tokens(req.mode)

    def _stream():
        for chunk in llm(
            _build_generate_prompt(req),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=cfg["top_p"],
            stop=_STOP_TOKENS,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    return StreamingResponse(_stream(), media_type="text/plain")
