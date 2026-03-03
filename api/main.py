from fastapi import FastAPI
import requests
import redis
import os
from prometheus_client import start_http_server, Counter, Histogram

app = FastAPI()

redis_client = redis.Redis(host="redis", port=6379)

REQUESTS = Counter("api_requests", "Total API requests")
LATENCY = Histogram("api_latency_seconds", "API latency")

start_http_server(9000)

def extract_features(prompt):
    return [
        len(prompt)/100,
        0.5,
        0.3,
        0.7,
        0.2
    ]

@app.post("/query")
def query(prompt: str):
    REQUESTS.inc()

    with LATENCY.time():
        features = extract_features(prompt)

        resp = requests.post(
            "http://router-service:8002/route",
            json={"features": features}
        )

        print("Router status:", resp.status_code)
        print("Router body:", resp.text)

        route_resp = resp.json()
        action = route_resp.get("action")

        worker_resp = requests.post(
            "http://model-worker:8001/generate",
            json={"prompt": prompt}
        ).json()

        redis_client.lpush("logs", str(action))

        return {
            "action": action,
            "response": worker_resp["response"]
        }