"""
routers/huggingface.py
----------------------
Hugging Face Inference API endpoints (free serverless-compatible).

Requires:
  HUGGINGFACE_API_KEY=hf_... in .env
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore[assignment]

router = APIRouter(prefix="/huggingface", tags=["huggingface"])

_DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
_DEFAULT_MAX_TOKENS = int(os.getenv("HUGGINGFACE_MAX_TOKENS", 256))
_DEFAULT_TEMPERATURE = float(os.getenv("HUGGINGFACE_TEMPERATURE", 0.7))


class HFAskRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text")
    model: Optional[str] = Field(default=None, description=f"Model id (default: {_DEFAULT_MODEL})")
    max_new_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)


def _get_client() -> "InferenceClient":  # type: ignore[name-defined]
    if InferenceClient is None:
        raise HTTPException(status_code=500, detail="huggingface_hub is not installed. Run: pip install huggingface_hub")
    token = os.getenv("HUGGINGFACE_API_KEY", "")
    if not token:
        raise HTTPException(
            status_code=401,
            detail="HUGGINGFACE_API_KEY is not set. Add it to your .env file.",
        )
    return InferenceClient(token=token)


@router.post("/ask")
def huggingface_ask(req: HFAskRequest):
    """Single prompt inference via Hugging Face serverless API."""
    client = _get_client()
    model = req.model or _DEFAULT_MODEL
    max_tokens = req.max_new_tokens or _DEFAULT_MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else _DEFAULT_TEMPERATURE

    try:
        output = client.text_generation(
            prompt=req.prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "response": output,
        "model": model,
    }
