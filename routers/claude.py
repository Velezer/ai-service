"""
routers/claude.py
-----------------
Anthropic Claude endpoints via the Anthropic Python SDK.

Free tier: https://www.anthropic.com/pricing
  - claude-haiku-3-5  → fastest & cheapest; free-tier credits on sign-up
  - claude-3-haiku-20240307 → legacy fast model

Requires:
  ANTHROPIC_API_KEY=sk-ant-...   in .env  (https://console.anthropic.com/settings/keys)
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

router = APIRouter(prefix="/claude", tags=["claude"])

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
_DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", 512))
_DEFAULT_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", 0.7))


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_client() -> "anthropic.Anthropic":  # type: ignore[name-defined]
    if not _ANTHROPIC_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="anthropic package is not installed. Run: pip install anthropic",
        )
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="ANTHROPIC_API_KEY is not set. Add it to your .env file. "
                   "Get a free key at https://console.anthropic.com/settings/keys",
        )
    return anthropic.Anthropic(api_key=api_key)


def _resolve(model, max_tokens, temperature):
    return (
        model or _DEFAULT_MODEL,
        max_tokens or _DEFAULT_MAX_TOKENS,
        temperature if temperature is not None else _DEFAULT_TEMPERATURE,
    )


# ── request schemas ───────────────────────────────────────────────────────────

class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(default="user")
    content: str


class ClaudeChatRequest(BaseModel):
    messages: list[ClaudeMessage] = Field(..., description="Conversation history (alternating user/assistant)")
    system: Optional[str] = Field(default=None, description="Optional system prompt")
    model: Optional[str] = Field(
        default=None,
        description=f"Claude model (default: {_DEFAULT_MODEL}). "
                    "Options: claude-3-haiku-20240307, claude-3-5-haiku-20241022, "
                    "claude-3-5-sonnet-20241022",
    )
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


class ClaudeAskRequest(BaseModel):
    prompt: str = Field(..., description="A single user message")
    system: Optional[str] = Field(default=None, description="Optional system prompt")
    model: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask")
def claude_ask(req: ClaudeAskRequest):
    """Send a single prompt to Claude and get a reply."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": req.prompt}],
    )
    if req.system:
        kwargs["system"] = req.system

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "response": response.content[0].text,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


@router.post("/chat")
def claude_chat(req: ClaudeChatRequest):
    """Multi-turn conversation with Claude."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
    )
    if req.system:
        kwargs["system"] = req.system

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "response": response.content[0].text,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


@router.post("/stream")
def claude_stream(req: ClaudeAskRequest):
    """Streaming single-prompt response from Claude."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)

    def _stream():
        try:
            kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": req.prompt}],
            )
            if req.system:
                kwargs["system"] = req.system

            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(_stream(), media_type="text/plain")
