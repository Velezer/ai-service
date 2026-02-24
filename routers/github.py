"""
routers/github.py
-----------------
GitHub-specific endpoints: PR descriptions, issue titles, release notes,
review comments, and label suggestions.
Model: Phi-2 (Q4_K_M) — same binary as git domain, shared model path by default.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from model_registry import get_model, get_cfg

router = APIRouter(prefix="/github", tags=["github"])
DOMAIN = "github"


# ── request schemas ───────────────────────────────────────────────────────────

class PRDescriptionRequest(BaseModel):
    title: str = Field(..., description="PR title")
    diff_summary: str = Field(..., description="Summary of changes (or raw diff)")
    issue_ref: Optional[str] = Field(default=None, description="Related issue number or URL")
    style: Literal["concise", "detailed"] = Field(default="concise")


class IssueTitleRequest(BaseModel):
    description: str = Field(..., description="Describe the bug or feature request")
    type: Literal["bug", "feature", "chore", "docs"] = Field(default="bug")


class ReleaseNotesRequest(BaseModel):
    version: str = Field(..., description="Version tag e.g. v1.2.0")
    commits: list[str] = Field(..., description="List of commit messages since last release")
    audience: Literal["technical", "user-facing"] = Field(default="technical")


class ReviewCommentRequest(BaseModel):
    code_snippet: str = Field(..., description="Code being reviewed")
    concern: str = Field(..., description="What to comment on")
    tone: Literal["constructive", "blocking", "nit"] = Field(default="constructive")


class LabelSuggestionRequest(BaseModel):
    title: str = Field(..., description="Issue or PR title")
    body: str = Field(..., description="Issue or PR body text")
    available_labels: list[str] = Field(..., description="Labels available in the repo")


# ── prompt builders ───────────────────────────────────────────────────────────

def _build_pr_prompt(req: PRDescriptionRequest) -> str:
    issue = f"\nCloses: {req.issue_ref}" if req.issue_ref else ""
    detail = (
        "Write a detailed PR description with ## What, ## Why, and ## How sections."
        if req.style == "detailed"
        else "Write a concise 2-3 sentence PR description."
    )
    return (
        f"You are a GitHub expert. {detail}\n"
        f"PR title: {req.title.strip()}\n"
        f"Changes: {req.diff_summary.strip()}"
        f"{issue}\n\n"
        "PR description:"
    )


def _build_issue_title_prompt(req: IssueTitleRequest) -> str:
    return (
        f"Write a clear, concise GitHub issue title for a {req.type} report.\n"
        "Output only the title — no labels, no markdown.\n"
        f"Description: {req.description.strip()}\n"
        "Title:"
    )


def _build_release_notes_prompt(req: ReleaseNotesRequest) -> str:
    commit_list = "\n".join(f"  - {c}" for c in req.commits)
    audience_hint = (
        "Use technical language, include breaking changes and migration notes."
        if req.audience == "technical"
        else "Use plain language suitable for end users. Focus on user-visible changes."
    )
    return (
        f"Write release notes for {req.version}. {audience_hint}\n"
        f"Commits:\n{commit_list}\n\n"
        "Release notes:"
    )


def _build_review_comment_prompt(req: ReviewCommentRequest) -> str:
    tone_hint = {
        "constructive": "Be helpful and suggest an improvement.",
        "blocking": "Clearly state why this must be changed before merging.",
        "nit": "Mark as a minor nit — optional to fix.",
    }[req.tone]
    return (
        f"Write a GitHub PR review comment. {tone_hint}\n"
        f"Concern: {req.concern.strip()}\n\n"
        f"Code:\n```\n{req.code_snippet.strip()}\n```\n\n"
        "Review comment:"
    )


def _build_label_prompt(req: LabelSuggestionRequest) -> str:
    labels = ", ".join(req.available_labels)
    return (
        "Choose the most appropriate labels from the list for this GitHub issue/PR.\n"
        "Return only a JSON array of label strings.\n"
        f"Available labels: {labels}\n"
        f"Title: {req.title.strip()}\n"
        f"Body: {req.body.strip()[:400]}\n"
        "Labels:"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _cfg():
    return get_cfg(DOMAIN)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/pr-description")
def github_pr_description(req: PRDescriptionRequest):
    """Generate a GitHub PR description."""
    llm = get_model(DOMAIN)
    cfg = _cfg()
    output = llm(
        _build_pr_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n\n\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip()}


@router.post("/issue-title")
def github_issue_title(req: IssueTitleRequest):
    """Generate a GitHub issue title."""
    llm = get_model(DOMAIN)
    cfg = _cfg()
    output = llm(
        _build_issue_title_prompt(req),
        max_tokens=48,
        temperature=min(cfg["temperature"], 0.2),
        top_p=cfg["top_p"],
        stop=["\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "type": req.type}


@router.post("/release-notes")
def github_release_notes(req: ReleaseNotesRequest):
    """Generate release notes from a list of commits."""
    llm = get_model(DOMAIN)
    cfg = _cfg()
    output = llm(
        _build_release_notes_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n\n\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "version": req.version}


@router.post("/review-comment")
def github_review_comment(req: ReviewCommentRequest):
    """Generate a PR review comment."""
    llm = get_model(DOMAIN)
    cfg = _cfg()
    output = llm(
        _build_review_comment_prompt(req),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["\n\n\n"],
        echo=False,
    )
    return {"response": output["choices"][0]["text"].strip(), "tone": req.tone}


@router.post("/label-suggestion")
def github_label_suggestion(req: LabelSuggestionRequest):
    """Suggest labels for a GitHub issue or PR."""
    llm = get_model(DOMAIN)
    cfg = _cfg()
    output = llm(
        _build_label_prompt(req),
        max_tokens=64,
        temperature=min(cfg["temperature"], 0.1),
        top_p=cfg["top_p"],
        stop=["\n\n", "]"],
        echo=False,
    )
    raw = output["choices"][0]["text"].strip()
    # Ensure the JSON array is closed if the model stopped early
    if raw and not raw.endswith("]"):
        raw = raw + "]"
    return {"response": raw}


@router.post("/stream/pr-description")
def github_stream_pr(req: PRDescriptionRequest):
    """Streaming PR description generation."""
    llm = get_model(DOMAIN)
    cfg = _cfg()

    def _stream():
        for chunk in llm(
            _build_pr_prompt(req),
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    return StreamingResponse(_stream(), media_type="text/plain")
