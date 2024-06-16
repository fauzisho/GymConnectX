# Import:
# -------
import random

import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(no_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, observation, epsilon):
        a = self.forward(observation)
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return a.argmax().item()
