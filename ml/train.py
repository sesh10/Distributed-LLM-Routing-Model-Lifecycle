import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import RouterModel
import mlflow

INPUT_DIM = 5
NUM_ACTIONS = 3

def one_hot(actions):
    return torch.eye(NUM_ACTIONS)[actions]

def generate_data(n=1000):
    X = np.random.rand(n, INPUT_DIM)
    actions = np.random.randint(0, NUM_ACTIONS, n)
    rewards = np.random.rand(n)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(actions),
        torch.tensor(rewards, dtype=torch.float32)
    )

mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run():
    X, actions, rewards = generate_data()
    model = RouterModel()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        preds = model(X, one_hot(actions)).squeeze()
        loss = loss_fn(preds, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mlflow.log_metric("loss", loss.item(), step=epoch)

    torch.save(model.state_dict(), "/shared/router.pt")
    mlflow.log_artifact("/shared/router.pt")