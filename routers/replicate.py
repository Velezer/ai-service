"""
routers/replicate.py
--------------------
Replicate API endpoints (free credits available on new accounts).

Requires:
  REPLICATE_API_TOKEN=r8_... in .env
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import replicate
except ImportError:
    replicate = None  # type: ignore[assignment]

router = APIRouter(prefix="/replicate", tags=["replicate"])

_DEFAULT_MODEL = os.getenv("REPLICATE_MODEL", "meta/meta-llama-3-8b-instruct")


class ReplicateAskRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for the selected Replicate model")
    model: Optional[str] = Field(default=None, description=f"Replicate model slug (default: {_DEFAULT_MODEL})")
    input: Optional[dict[str, Any]] = Field(default=None, description="Additional model-specific input fields")


def _require_client() -> None:
    if replicate is None:
        raise HTTPException(status_code=500, detail="replicate package is not installed. Run: pip install replicate")
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(status_code=401, detail="REPLICATE_API_TOKEN is not set. Add it to your .env file.")


@router.post("/ask")
def replicate_ask(req: ReplicateAskRequest):
    """Run a text prompt against a Replicate model."""
    _require_client()
    model = req.model or _DEFAULT_MODEL
    payload = {"prompt": req.prompt}
    if req.input:
        payload.update(req.input)

    try:
        output = replicate.run(model, input=payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if isinstance(output, str):
        text = output
    else:
        text = "".join(str(part) for part in output)

    return {
        "response": text,
        "model": model,
    }
