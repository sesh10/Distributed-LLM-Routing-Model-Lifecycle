FROM base-ml

WORKDIR /app

COPY ml/ ml/
COPY retrainer/main.py .

RUN pip install mlflow-skinny pandas
RUN apt-get update && apt-get install -y git

CMD ["python", "main.py"]