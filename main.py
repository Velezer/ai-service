import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from typing import Literal


load_dotenv()

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH")
N_CTX = int(os.getenv("N_CTX", 4096))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 256))
N_THREADS = int(os.getenv("N_THREADS", 1))
N_BATCH = int(os.getenv("N_BATCH", 256))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
TOP_P = float(os.getenv("TOP_P", 0.95))
USE_MLOCK = os.getenv("USE_MLOCK", "true").lower() in {"1", "true", "yes"}
FAST_MAX_TOKENS = int(os.getenv("FAST_MAX_TOKENS", 96))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    use_mlock=USE_MLOCK,
)


class GenerateRequest(BaseModel):
    prompt: str


class GenerateCodeRequest(BaseModel):
    instruction: str = Field(..., description="Describe the code you want generated")
    language: str = Field(default="python", description="Target programming language")
    context: str | None = Field(default=None, description="Optional context and constraints")
    mode: Literal["quality", "fast"] = Field(default="quality", description="Generation mode")


class GenerateRustFastRequest(BaseModel):
    instruction: str = Field(..., description="Describe the Rust code you want generated")
    context: str | None = Field(default=None, description="Optional project constraints")


def _build_code_prompt(req: GenerateCodeRequest) -> str:
    context = f"\nProject context and constraints:\n{req.context.strip()}\n" if req.context else ""
    return (
        "You are a senior software engineer. "
        "Generate production-quality code and keep your answer focused on code output.\n"
        f"Target language: {req.language}\n"
        f"Task: {req.instruction.strip()}"
        f"{context}\n"
        "Output:")


def _build_rust_fast_prompt(instruction: str, context: str | None) -> str:
    fast_context = f"\nConstraints:\n{context.strip()}\n" if context else ""
    return (
        "Return only valid Rust code. No explanations. "
        "Prefer std library and short implementations.\n"
        f"Task: {instruction.strip()}"
        f"{fast_context}\n"
        "Rust code:")


def _generation_settings(mode: str) -> tuple[int, float]:
    if mode == "fast":
        return FAST_MAX_TOKENS, min(TEMPERATURE, 0.15)
    return MAX_TOKENS, min(TEMPERATURE, 0.25)


@app.post("/generate/")
def generate_text(req: GenerateRequest):
    output = llm(
        req.prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        echo=False,
    )

    return {"response": output["choices"][0]["text"]}


@app.post("/generate/code")
def generate_code(req: GenerateCodeRequest):
    prompt = _build_code_prompt(req)
    max_tokens, temperature = _generation_settings(req.mode)
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=TOP_P,
        stop=["\n\n```", "\n\nExplanation:", "\n\nNotes:"],
        echo=False,
    )

    return {
        "response": output["choices"][0]["text"],
        "language": req.language,
    }


@app.post("/generate/code/rust/fast")
def generate_rust_fast(req: GenerateRustFastRequest):
    prompt = _build_rust_fast_prompt(req.instruction, req.context)
    output = llm(
        prompt,
        max_tokens=FAST_MAX_TOKENS,
        temperature=min(TEMPERATURE, 0.1),
        top_p=min(TOP_P, 0.9),
        stop=["\n\n```", "\n\nExplanation:", "\n\nNotes:"],
        echo=False,
    )

    return {
        "response": output["choices"][0]["text"],
        "language": "rust",
        "mode": "fast",
    }


@app.post("/generate/stream/plain")
def generate_stream_plain(req: GenerateRequest):

    def stream():
        for chunk in llm(
            req.prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/generate/stream/event")
def generate_stream_event(req: GenerateRequest):

    def stream():
        for chunk in llm(
            req.prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield f"data: {token}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
