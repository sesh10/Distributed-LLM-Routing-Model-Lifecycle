## LLM Routing Demo Project

This repository contains a small, containerized LLM routing demo. It shows how to:

- **Expose a simple API** that accepts prompts.
- **Route requests** using a learned bandit-style router.
- **Serve generations** from a Hugging Face model worker.
- **Train and retrain** the router model offline.
- **Monitor the system** with Prometheus and track experiments with MLflow.

All components are wired together via `docker-compose`.

---

## Architecture Overview

- **`api` service**
  - FastAPI application (`api/main.py`).
  - Accepts a prompt on `POST /query`.
  - Extracts simple numeric features from the prompt.
  - Calls the **router service** to choose an action.
  - Calls the **model-worker** to generate text.
  - Logs chosen actions to Redis and exposes Prometheus metrics on port `9000`.

- **`router-service`**
  - FastAPI application (`router-service/main.py`).
  - Uses a PyTorch `RouterModel` (`router-service/model.py`) and a `thompson_sampling` policy (`router-service/policy.py`).
  - Loads model weights from `/shared/router.pt` if available.
  - Endpoint `POST /route` takes a list of 5 features and returns an `action` (integer).
  - Exposes Prometheus metrics on port `9001`.

- **`model-worker`**
  - FastAPI application (`model-worker/main.py`).
  - Uses `transformers.pipeline("text-generation", model="EleutherAI/gpt-neo-125M")`.
  - Endpoint `POST /generate` takes a `prompt` and returns generated text.
  - Exposes Prometheus metrics on port `9002`.

- **`ml` (training)**
  - Contains `ml/model.py` with the `RouterModel` definition.
  - `ml/train.py`:
    - Generates synthetic data (features, actions, rewards).
    - Trains the router model with PyTorch.
    - Logs metrics and artifacts to **MLflow**.
    - Saves the trained model to `/shared/router.pt`.

- **`retrainer`**
  - Service/Dockerfile intended to run retraining flows using the shared volume.

- **Infrastructure / supporting services**
  - **Redis**: used by the API to log actions.
  - **MLflow** (`mlflow` service): experiment tracking UI on port `5000`.
  - **Prometheus** (`prometheus` service): metrics scraping, configured via `monitoring/prometheus.yml`, UI on port `9090`.
  - **`base-ml`**: base image used by ML-related services.
  - All services share a Docker volume named `shared` for model artifacts (e.g. `router.pt`).

The top-level `docker-compose.yaml` defines and wires all these services together.

---

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** installed.
- Sufficient memory to load the `EleutherAI/gpt-neo-125M` model (a few GB of RAM is recommended).
- Internet access on first run so that the Hugging Face model can be downloaded.

### Clone the Repository

```bash
git clone <your-repo-url> llmproject
cd llmproject
```

---

## Running the Stack

From the repository root:

```bash
docker compose up --build
```

This will:

- Build the `base-ml`, `api`, `router-service`, `model-worker`, and `retrainer` images.
- Start Redis, MLflow, and Prometheus.
- Create the `shared` volume for model artifacts.

Once all containers are healthy:

- **API**: `http://localhost:8000`
- **MLflow UI**: `http://localhost:5000`
- **Prometheus UI**: `http://localhost:9090`

Stop everything with:

```bash
docker compose down
```

---

## Training the Router Model

The router uses a PyTorch model trained with the script in `ml/train.py`.

When run (typically inside a container that shares the `shared` volume), it will:

- Generate synthetic features, actions, and rewards.
- Train `RouterModel` on this data.
- Log training **loss** to MLflow.
- Save the trained weights to `/shared/router.pt`.

Once `/shared/router.pt` exists, `router-service` will automatically load it at startup:

- If the file is present: `Loaded trained router model.`
- If missing: `No trained model found. Using randomly initialized model.`

Depending on your workflow, training can be triggered via:

- A one-off container based on the `base-ml` or `retrainer` image.
- A job that runs `python ml/train.py` with `/shared` mounted.

---

## API Usage

### 1. Query Endpoint (Front API)

- **URL**: `POST http://localhost:8000/query`
- **Body** (JSON-encoded string or a simple form field; current implementation expects `prompt: str` directly):

Example using `curl` with JSON:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '"Tell me a story about space exploration."'
```

The API will:

- Extract simple numeric features from the prompt.
- Ask the router for an `action`.
- Ask the model worker for generated text.
- Push the chosen `action` into Redis list `logs`.

**Response shape**:

```json
{
  "action": <int>,
  "response": "<generated text>"
}
```

### 2. Router Endpoint (Internal)

- **URL**: `POST http://router-service:8002/route` (inside Docker network)
- **Body**:

```json
{
  "features": [0.1, 0.5, 0.3, 0.7, 0.2]
}
```

**Response**:

```json
{
  "action": 1
}
```

### 3. Model Worker Endpoint (Internal)

- **URL**: `POST http://model-worker:8001/generate`
- **Body**:

```json
{
  "prompt": "Tell me a story about space exploration."
}
```

**Response**:

```json
{
  "response": "Tell me a story about space exploration. ..."
}
```

---

## Monitoring and Observability

- **Prometheus metrics**:
  - API exports counters/histograms such as `api_requests`, `api_latency_seconds` on port `9000`.
  - Router exports `router_requests` on port `9001`.
  - Model worker exports `worker_requests` on port `9002`.
  - `monitoring/prometheus.yml` configures how Prometheus scrapes these endpoints.

- **MLflow**:
  - Training script logs:
    - `loss` per epoch.
    - The `router.pt` artifact.
  - Access the tracking UI at `http://localhost:5000` to inspect runs.

---

## Development Notes

- Python dependencies for each service live in their respective `requirements.txt`:
  - `api/requirements.txt`
  - `router-service/requirements.txt`
  - `model-worker/requirements.txt`
  - `retrainer/requirements.txt`
- Each major service has its own `Dockerfile`.
- The `shared` volume is critical for moving trained model weights between training and inference services.

---

## Future Improvements

Some ideas for extending this demo:

- Hook the router’s rewards to real user feedback or latency measurements instead of synthetic data.
- Add additional model workers and extend the router to handle more actions.
- Implement scheduled retraining jobs that periodically update `/shared/router.pt`.
- Harden input validation and error handling on public endpoints.

