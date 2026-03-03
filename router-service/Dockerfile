FROM base-ml

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn numpy prometheus-client python-multipart

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]