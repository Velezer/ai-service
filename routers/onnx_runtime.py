"""
routers/onnx_runtime.py
-----------------------
Generic ONNX Runtime inference endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

router = APIRouter(prefix="/onnx", tags=["onnx"])


class ONNXInferRequest(BaseModel):
    model_path: str = Field(..., description="Path to .onnx model file")
    inputs: dict[str, Any] = Field(..., description="Mapping: input_name -> nested list/array values")
    output_names: Optional[list[str]] = Field(default=None, description="Optional subset of outputs")


@router.post("/infer")
def onnx_infer(req: ONNXInferRequest):
    """Run inference for any ONNX model via CPUExecutionProvider."""
    if ort is None:
        raise HTTPException(status_code=500, detail="onnxruntime is not installed. Run: pip install onnxruntime")

    try:
        session = ort.InferenceSession(req.model_path, providers=["CPUExecutionProvider"])
        feed = {name: np.asarray(value) for name, value in req.inputs.items()}
        outputs = session.run(req.output_names, feed)
        resolved_names = req.output_names or [o.name for o in session.get_outputs()]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = {}
    for name, value in zip(resolved_names, outputs):
        result[name] = np.asarray(value).tolist()

    return {
        "outputs": result,
        "provider": "CPUExecutionProvider",
    }
