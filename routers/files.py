"""
routers/files.py
----------------
Basic file-operation endpoints.
Model: Qwen2.5-Coder-0.5B (Q4_K_M) — tiny, fast, good at file/path tasks.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from model_registry import get_model, get_cfg

router = APIRouter(prefix="/files", tags=["files"])
DOMAIN = "files"


# ── request schemas ───────────────────────────────────────────────────────────

class FileOpRequest(BaseModel):
    instruction: str = Field(..., description="Describe the file operation task")
    context: Optional[str] = Field(default=None, description="File content or path context")
    mode: Literal["quality", "fast"] = Field(default="quality")


class FileRenameRequest(BaseModel):
    files: list[str] = Field(..., description="List of current filenames")
    rule: str = Field(..., description="Renaming rule or pattern to apply")


class FileFilterRequest(BaseModel):
    files: list[str] = Field(..., description="List of filenames to filter")
    criteria: str = Field(..., description="Filter criteria (e.g. 'only .rs files', 'no test files')")


# ── prompt builders ───────────────────────────────────────────────────────────

def _build_file_prompt(req: FileOpRequest) -> str:
    ctx = f"\nFile content / context:\n{req.context.strip()}\n" if req.context else ""
    return (
        "You are a file-system assistant. Output only the result — no explanations.\n"
        f"Task: {req.instruction.strip()}"
        f"{ctx}\n"
        "Output:"
    )


def _build_rename_prompt(req: FileRenameRequest) -> str:
    file_list = "\n".join(f"  - {f}" for f in req.files)
    return (
        "You are a file-renaming assistant. "
        "Apply the rule to each filename and return a JSON object mapping old → new names.\n"
        f"Rule: {req.rule.strip()}\n"
        f"Files:\n{file_list}\n"
        "JSON output:"
    )


def _build_filter_prompt(req: FileFilterRequest) -> str:
    file_list = "\n".join(f"  - {f}" for f in req.files)
    return (
        "You are a file-filter assistant. "
        "Return only the filenames that match the criteria as a JSON array.\n"
        f"Criteria: {req.criteria.strip()}\n"
        f"Files:\n{file_list}\n"
        "JSON array:"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokens(mode: str) -> tuple[int, float]:
    cfg = get_cfg(DOMAIN)
    if mode == "fast":
        return max(64, cfg["max_tokens"] // 2), min(cfg["temperature"], 0.15)
    return cfg["max_tokens"], cfg["temperature"]


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate")
def file_generate(req: FileOpRequest):
    """General file-operation generation."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    max_tokens, temperature = _tokens(req.mode)
    output = llm(
        _build_file_prompt(req),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=cfg["top_p"],
        stop=["Explanation:", "Note:"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip()}


@router.post("/rename")
def file_rename(req: FileRenameRequest):
    """Suggest new filenames based on a renaming rule."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_rename_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip()}


@router.post("/filter")
def file_filter(req: FileFilterRequest):
    """Filter a list of filenames by criteria."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    output = llm(
        _build_filter_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip()}


@router.post("/stream")
def file_stream(req: FileOpRequest):
    """Streaming file-operation generation."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    max_tokens, temperature = _tokens(req.mode)

    def _stream():
        for chunk in llm(
            _build_file_prompt(req),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=cfg["top_p"],
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    return StreamingResponse(_stream(), media_type="text/plain")
