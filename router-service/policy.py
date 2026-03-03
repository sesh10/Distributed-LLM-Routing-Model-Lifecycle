import torch
import numpy as np

def thompson_sampling(model, x, num_actions=3):
    samples = []
    for action in range(num_actions):
        action_oh = torch.eye(num_actions)[action].unsqueeze(0)
        mean_reward = model(x, action_oh).item()
        sampled = np.random.normal(mean_reward, 0.1)
        samples.append(sampled)
    return int(np.argmax(samples))