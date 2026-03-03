import time
import subprocess

while True:
    print("Retraining model...")
    subprocess.run(["python", "/app/ml/train.py"])
    time.sleep(3600)