FROM python:3.10-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

COPY . .

RUN pip install fastapi uvicorn redis requests prometheus-client

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]