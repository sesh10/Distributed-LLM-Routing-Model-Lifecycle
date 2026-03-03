import torch
import torch.nn as nn


class RouterModel(nn.Module):
    """
    Contextual bandit reward predictor.

    Input:
        - features vector (batch_size, input_dim)
        - action one-hot vector (batch_size, num_actions)

    Output:
        - predicted reward (batch_size, 1)
    """

    def __init__(self, input_dim=5, num_actions=3):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(input_dim + num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features, action_onehot):
        """
        features: Tensor shape (batch_size, input_dim)
        action_onehot: Tensor shape (batch_size, num_actions)
        """
        x = torch.cat([features, action_onehot], dim=1)
        return self.network(x)