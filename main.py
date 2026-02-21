from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

LLAMA_BINARY = "./llama-cli"
MODEL_PATH = "./models/ggmlv3-pythia-70m-deduped-q5_1.bi"

@app.post("/generate/")
def generate(request: GenerateRequest):
    prompt = request.prompt

    result = subprocess.run(
        [LLAMA_BINARY, "-m", MODEL_PATH, "-p", prompt, "-n", "50"],
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }