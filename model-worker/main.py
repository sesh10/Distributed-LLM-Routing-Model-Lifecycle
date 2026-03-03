import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import start_http_server, Counter

class GenerateRequest(BaseModel):
    prompt: str

app = FastAPI()
REQUESTS = Counter("worker_requests", "Total inference requests")

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

start_http_server(9002)

@app.post("/generate")
def generate(req: GenerateRequest):
    REQUESTS.inc()
    result = generator(
        req.prompt,
        max_length=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return {"response": result[0]["generated_text"]}