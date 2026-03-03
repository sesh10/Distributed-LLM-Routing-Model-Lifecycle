from fastapi import FastAPI
import torch
import os
from pydantic import BaseModel
from typing import List
from model import RouterModel
from policy import thompson_sampling
from prometheus_client import start_http_server, Counter


class RouteRequest(BaseModel):
    features: List[float]

app = FastAPI()

REQUESTS = Counter("router_requests", "Total routing requests")

INPUT_DIM = 5
NUM_ACTIONS = 3

model = RouterModel(INPUT_DIM, NUM_ACTIONS)

MODEL_PATH = "/shared/router.pt"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print("Loaded trained router model.")
else:
    print("No trained model found. Using randomly initialized model.")

model.eval()

start_http_server(9001)


@app.post("/route")
def route(req: RouteRequest):
    REQUESTS.inc()
    features = req.features

    if len(features) != INPUT_DIM:
        return {"error": f"Expected {INPUT_DIM} features"}

    x = torch.tensor([features], dtype=torch.float32)
    action = thompson_sampling(model, x)

    return {"action": action}