FROM base-ml

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn transformers prometheus-client

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]