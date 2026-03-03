import torch
import torch.nn as nn

class RouterModel(nn.Module):
    def __init__(self, input_dim=5, num_actions=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim + num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features, action_onehot):
        x = torch.cat([features, action_onehot], dim=1)
        return self.network(x)