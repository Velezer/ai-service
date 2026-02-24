"""
routers/git.py
--------------
Git-specific endpoints: commit messages, diff summaries, branch names, rebases.
Model: Phi-2 (Q4_K_M) — strong at short structured text, fast on 1 vCPU.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from model_registry import get_model, get_cfg

router = APIRouter(prefix="/git", tags=["git"])
DOMAIN = "git"


# ── request schemas ───────────────────────────────────────────────────────────

class CommitMessageRequest(BaseModel):
    diff: str = Field(..., description="Git diff output (git diff --staged)")
    style: Literal["conventional", "short", "detailed"] = Field(
        default="conventional",
        description="Commit message style",
    )


class DiffSummaryRequest(BaseModel):
    diff: str = Field(..., description="Git diff to summarise")
    max_lines: int = Field(default=5, description="Max bullet points in summary")


class BranchNameRequest(BaseModel):
    description: str = Field(..., description="What the branch is for")
    prefix: Optional[str] = Field(default=None, description="Optional prefix e.g. feat/, fix/")


class GitCommandRequest(BaseModel):
    task: str = Field(..., description="Describe what you want to do with git")
    context: Optional[str] = Field(default=None, description="Repo state or constraints")


# ── prompt builders ───────────────────────────────────────────────────────────

_STYLE_HINT = {
    "conventional": "Use Conventional Commits format: <type>(<scope>): <subject>.",
    "short": "Write a single short imperative sentence (max 72 chars).",
    "detailed": "Write a subject line then a blank line then a body with bullet points.",
}


def _build_commit_prompt(req: CommitMessageRequest) -> str:
    return (
        "You are a git expert. Write a commit message for the following diff.\n"
        f"{_STYLE_HINT[req.style]}\n"
        "Output only the commit message — no extra text.\n\n"
        f"Diff:\n{req.diff.strip()}\n\n"
        "Commit message:"
    )


def _build_diff_summary_prompt(req: DiffSummaryRequest) -> str:
    return (
        "Summarise the following git diff as a concise bullet-point list "
        f"(max {req.max_lines} bullets). Be specific about what changed.\n\n"
        f"Diff:\n{req.diff.strip()}\n\n"
        "Summary:"
    )


def _build_branch_name_prompt(req: BranchNameRequest) -> str:
    prefix_hint = f"Start with '{req.prefix}'" if req.prefix else "Use a suitable prefix (feat/, fix/, chore/, etc.)"
    return (
        "Generate a concise git branch name (kebab-case, max 40 chars).\n"
        f"{prefix_hint}\n"
        f"Purpose: {req.description.strip()}\n"
        "Branch name:"
    )


def _build_git_command_prompt(req: GitCommandRequest) -> str:
    ctx = f"\nContext:\n{req.context.strip()}\n" if req.context else ""
    return (
        "You are a git expert. Output only the exact git command(s) needed — no explanations.\n"
        f"Task: {req.task.strip()}"
        f"{ctx}\n"
        "Command:"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokens(mode: str = "quality") -> tuple[int, float]:
    cfg = get_cfg(DOMAIN)
    if mode == "fast":
        return max(48, cfg["max_tokens"] // 2), min(cfg["temperature"], 0.15)
    return cfg["max_tokens"], cfg["temperature"]


def _generate_text(
    prompt: str,
    *,
    stop: list[str] | None = None,
    mode: str = "quality",
    max_tokens: int | None = None,
) -> str:
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)
    token_limit, temperature = _tokens(mode)
    output = llm(
        prompt,
        max_tokens=max_tokens if max_tokens is not None else token_limit,
        temperature=temperature,
        top_p=cfg["top_p"],
        stop=stop,
        echo=False,
    )
    return output["choices"][0]["text"].strip()


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/commit-message")
def git_commit_message(req: CommitMessageRequest):
    """Generate a commit message from a git diff."""
    response = _generate_text(_build_commit_prompt(req), stop=["Diff:", "---"])
    return {"response": response, "style": req.style}


@router.post("/diff-summary")
def git_diff_summary(req: DiffSummaryRequest):
    """Summarise a git diff as bullet points."""
    response = _generate_text(_build_diff_summary_prompt(req), stop=["\n\n\n"])
    return {"response": response}


@router.post("/branch-name")
def git_branch_name(req: BranchNameRequest):
    """Suggest a git branch name."""
    response = _generate_text(_build_branch_name_prompt(req), stop=["\n", " "], mode="fast", max_tokens=48)
    return {"response": response}


@router.post("/command")
def git_command(req: GitCommandRequest):
    """Suggest the git command(s) for a described task."""
    response = _generate_text(_build_git_command_prompt(req), stop=["Explanation:"])
    return {"response": response}


@router.post("/stream/commit-message")
def git_stream_commit(req: CommitMessageRequest):
    """Streaming commit message generation."""
    llm = get_model(DOMAIN)
    cfg = get_cfg(DOMAIN)

    def _stream():
        for chunk in llm(
            _build_commit_prompt(req),
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    return StreamingResponse(_stream(), media_type="text/plain")
