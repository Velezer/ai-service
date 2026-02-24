"""
routers/groq_chat.py
--------------------
Groq Cloud endpoints — extremely fast inference on LPU hardware.

Free tier: https://console.groq.com/docs/rate-limits
  - llama-3.3-70b-versatile  → 30 RPM / 6 000 TPM / 14 400 RPD  (free)
  - llama-3.1-8b-instant     → 30 RPM / 20 000 TPM / 14 400 RPD (free, fastest)
  - mixtral-8x7b-32768       → 30 RPM / 5 000 TPM / 14 400 RPD  (free)
  - gemma2-9b-it             → 30 RPM / 15 000 TPM / 14 400 RPD (free)

Requires:
  GROQ_API_KEY=gsk_...   in .env  (https://console.groq.com/keys)
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq, GroqError
    _GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    GroqError = Exception  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False

router = APIRouter(prefix="/groq", tags=["groq"])

_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 512))
_DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", 0.7))


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_client() -> "Groq":  # type: ignore[name-defined]
    if not _GROQ_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="groq package is not installed. Run: pip install groq",
        )
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="GROQ_API_KEY is not set. Add it to your .env file. Get a free key at https://console.groq.com/keys",
        )
    return Groq(api_key=api_key)


def _resolve(model, max_tokens, temperature):
    return (
        model or _DEFAULT_MODEL,
        max_tokens or _DEFAULT_MAX_TOKENS,
        temperature if temperature is not None else _DEFAULT_TEMPERATURE,
    )


# ── request schemas ───────────────────────────────────────────────────────────

class GroqMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str


class GroqChatRequest(BaseModel):
    messages: list[GroqMessage] = Field(..., description="Conversation history")
    model: Optional[str] = Field(
        default=None,
        description=f"Groq model ID (default: {_DEFAULT_MODEL}). "
                    "Options: llama-3.3-70b-versatile, llama-3.1-8b-instant, "
                    "mixtral-8x7b-32768, gemma2-9b-it",
    )
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


class GroqAskRequest(BaseModel):
    prompt: str = Field(..., description="A single user message")
    system: Optional[str] = Field(default=None, description="Optional system prompt")
    model: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask")
def groq_ask(req: GroqAskRequest):
    """Send a single prompt to Groq and get an ultra-fast reply."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except GroqError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    choice = response.choices[0]
    return {
        "response": choice.message.content,
        "model": response.model,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


@router.post("/chat")
def groq_chat(req: GroqChatRequest):
    """Multi-turn conversation via Groq (OpenAI-compatible interface)."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except GroqError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    choice = response.choices[0]
    return {
        "response": choice.message.content,
        "model": response.model,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


@router.post("/stream")
def groq_stream(req: GroqAskRequest):
    """Streaming single-prompt response from Groq."""
    client = _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    def _stream():
        try:
            with client.chat.completions.stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except GroqError as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(_stream(), media_type="text/plain")
