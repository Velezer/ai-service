"""
routers/gemini.py
-----------------
Google Gemini endpoints via the Google Generative AI SDK.

Free tier: https://ai.google.dev/pricing
  - gemini-1.5-flash  → 15 RPM / 1 M TPM / 1 500 RPD  (free)
  - gemini-1.5-pro    → 2 RPM / 32 k TPM / 50 RPD     (free, slower)

Requires:
  GEMINI_API_KEY=AIza...   in .env  (https://aistudio.google.com/app/apikey)
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False

router = APIRouter(prefix="/gemini", tags=["gemini"])

_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_DEFAULT_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 512))
_DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.7))


# ── helpers ───────────────────────────────────────────────────────────────────

def _require():
    if not _GENAI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="google-generativeai package is not installed. Run: pip install google-generativeai",
        )
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="GEMINI_API_KEY is not set. Add it to your .env file. Get a free key at https://aistudio.google.com/app/apikey",
        )
    genai.configure(api_key=api_key)


def _get_model(model: Optional[str]) -> "genai.GenerativeModel":  # type: ignore[name-defined]
    return genai.GenerativeModel(model or _DEFAULT_MODEL)


def _gen_config(max_tokens: Optional[int], temperature: Optional[float]):
    return genai.types.GenerationConfig(  # type: ignore[attr-defined]
        max_output_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
        temperature=temperature if temperature is not None else _DEFAULT_TEMPERATURE,
    )


# ── request schemas ───────────────────────────────────────────────────────────

class GeminiAskRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to send to Gemini")
    system: Optional[str] = Field(default=None, description="Optional system instruction")
    model: Optional[str] = Field(default=None, description=f"Gemini model (default: {_DEFAULT_MODEL})")
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


class GeminiChatMessage(BaseModel):
    role: Literal["user", "model"] = Field(default="user")
    content: str


class GeminiChatRequest(BaseModel):
    messages: list[GeminiChatMessage] = Field(..., description="Conversation history")
    model: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask")
def gemini_ask(req: GeminiAskRequest):
    """Send a single prompt to Google Gemini and get a reply."""
    _require()
    mdl = _get_model(req.model)
    cfg = _gen_config(req.max_tokens, req.temperature)
    try:
        if req.system:
            # Prepend system instruction as a user turn (Gemini 1.5 supports system_instruction)
            mdl = genai.GenerativeModel(
                req.model or _DEFAULT_MODEL,
                system_instruction=req.system,
            )
        response = mdl.generate_content(req.prompt, generation_config=cfg)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "response": response.text,
        "model": req.model or _DEFAULT_MODEL,
        "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
    }


@router.post("/chat")
def gemini_chat(req: GeminiChatRequest):
    """Multi-turn conversation with Google Gemini."""
    _require()
    mdl = _get_model(req.model)
    cfg = _gen_config(req.max_tokens, req.temperature)

    # Build history (all but the last message) and send the last as the new turn
    history = [
        {"role": m.role, "parts": [m.content]}
        for m in req.messages[:-1]
    ]
    last = req.messages[-1].content

    try:
        chat = mdl.start_chat(history=history)
        response = chat.send_message(last, generation_config=cfg)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "response": response.text,
        "model": req.model or _DEFAULT_MODEL,
    }


@router.post("/stream")
def gemini_stream(req: GeminiAskRequest):
    """Streaming single-prompt response from Google Gemini."""
    _require()
    cfg = _gen_config(req.max_tokens, req.temperature)

    def _stream():
        try:
            mdl = _get_model(req.model)
            if req.system:
                mdl = genai.GenerativeModel(
                    req.model or _DEFAULT_MODEL,
                    system_instruction=req.system,
                )
            for chunk in mdl.generate_content(req.prompt, generation_config=cfg, stream=True):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(_stream(), media_type="text/plain")
