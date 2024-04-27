import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # Unpack the list of input dimensions for lunar lander env
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
    
    # Perform forward prop and activate using relu
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # We don't active via sigmoid or such since we need this raw q-value estimates to calculate MSE Loss
        # Sometimes, estimates can be negative so activating is incorrect.
        actions = self.fc3(x)
        return actions