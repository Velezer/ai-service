from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

llm = Llama(
    model_path="./models/pythia-70m-deduped-v0.f16.gguf",
    n_ctx=512
)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate(req: GenerateRequest):
    output = llm(
        req.prompt,
        max_tokens=64
    )

    return {
        "response": output["choices"][0]["text"]
    }