from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

LLAMA_BINARY = "./llama/llama-cli"
MODEL_PATH = "./models/ggml-pythia-70m-deduped-q4_0.bin"


class Request(BaseModel):
    prompt: str


@app.post("/generate/")
def generate(request: Request):
    prompt = request.prompt

    # Call llama.cpp executable
    try:
        result = subprocess.run(
            [LLAMA_BINARY, "-m", MODEL_PATH, "-p", prompt, "-n", "50"],
            capture_output=True,
            text=True
        )
        output_text = result.stdout
    except subprocess.CalledProcessError as e:
        output_text = f"Error: {e.stderr}"

    return {"response": output_text}
