import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse


load_dotenv()

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH")
N_CTX = int(os.getenv("N_CTX", 256))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 48))
N_THREADS = int(os.getenv("N_THREADS", 1))
N_BATCH = int(os.getenv("N_BATCH", 128))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_P = float(os.getenv("TOP_P", 0.9))
USE_MLOCK = bool(os.getenv("USE_MLOCK", True))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    use_mlock=USE_MLOCK
)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate(req: GenerateRequest):
    output = llm(
        req.prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        echo=False
    )

    return {
        "response": output["choices"][0]["text"]
    }

@app.post("/generate/stream/plain")
def generate(req: GenerateRequest):

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
def generate(req: GenerateRequest):

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