"""
routers/tflite_runtime.py
-------------------------
Generic TensorFlow Lite inference endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except ImportError:
        Interpreter = None  # type: ignore[assignment]

router = APIRouter(prefix="/tflite", tags=["tflite"])


class TFLiteInferRequest(BaseModel):
    model_path: str = Field(..., description="Path to .tflite model file")
    inputs: dict[str, Any] = Field(..., description="Mapping: input_tensor_name -> nested list/array")


@router.post("/infer")
def tflite_infer(req: TFLiteInferRequest):
    """Run inference for any TFLite model."""
    if Interpreter is None:
        raise HTTPException(
            status_code=500,
            detail="Neither tflite-runtime nor tensorflow is installed. Run: pip install tflite-runtime",
        )

    try:
        interpreter = Interpreter(model_path=req.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for detail in input_details:
            name = detail["name"]
            if name not in req.inputs:
                raise ValueError(f"Missing input tensor '{name}'")
            value = np.asarray(req.inputs[name], dtype=detail["dtype"])
            interpreter.set_tensor(detail["index"], value)

        interpreter.invoke()

        outputs = {
            detail["name"]: np.asarray(interpreter.get_tensor(detail["index"])).tolist()
            for detail in output_details
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"outputs": outputs}
