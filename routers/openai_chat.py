"""
routers/openai_chat.py
----------------------
ChatGPT endpoints via the OpenAI API.

Uses gpt-4o-mini by default — the cheapest OpenAI model, available on the
free-tier credit grant.  Set OPENAI_MODEL in .env to override (e.g. gpt-3.5-turbo).

Requires:
  OPENAI_API_KEY=sk-...   in .env  (https://platform.openai.com/api-keys)
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI, OpenAIError
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except ImportError:
    _client = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[misc,assignment]

router = APIRouter(prefix="/chatgpt", tags=["chatgpt"])

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 512))
_DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.7))


# ── request schemas ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user",
        description="Message role",
    )
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Conversation history (at least one user message)",
    )
    model: Optional[str] = Field(
        default=None,
        description=f"OpenAI model to use (default: {_DEFAULT_MODEL})",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description=f"Max tokens in the response (default: {_DEFAULT_MAX_TOKENS})",
    )
    temperature: Optional[float] = Field(
        default=None,
        description=f"Sampling temperature 0-2 (default: {_DEFAULT_TEMPERATURE})",
    )


class SimpleAskRequest(BaseModel):
    prompt: str = Field(..., description="A single user message to send to ChatGPT")
    system: Optional[str] = Field(
        default=None,
        description="Optional system prompt to set the assistant's behaviour",
    )
    model: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_client():
    if _client is None:
        raise HTTPException(
            status_code=500,
            detail="openai package is not installed. Run: pip install openai",
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="OPENAI_API_KEY is not set. Add it to your .env file.",
        )


def _resolve(model, max_tokens, temperature):
    return (
        model or _DEFAULT_MODEL,
        max_tokens or _DEFAULT_MAX_TOKENS,
        temperature if temperature is not None else _DEFAULT_TEMPERATURE,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat")
def chatgpt_chat(req: ChatRequest):
    """
    Send a full conversation history to ChatGPT and get a reply.

    Pass a list of messages with roles 'system', 'user', or 'assistant'
    to maintain multi-turn context.
    """
    _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except OpenAIError as exc:
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


@router.post("/ask")
def chatgpt_ask(req: SimpleAskRequest):
    """
    Send a single prompt to ChatGPT and get a reply.

    Simpler alternative to /chatgpt/chat — no need to build a message list.
    """
    _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except OpenAIError as exc:
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
def chatgpt_stream(req: SimpleAskRequest):
    """
    Streaming version of /chatgpt/ask — tokens are returned as they are generated.
    """
    _require_client()
    model, max_tokens, temperature = _resolve(req.model, req.max_tokens, req.temperature)
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    def _stream():
        try:
            with _client.chat.completions.stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except OpenAIError as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(_stream(), media_type="text/plain")
