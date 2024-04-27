import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from DeepQNetwork import *

class Agent():
    def __init__(self, discount_factor, epsilon, lr, input_dims, batch_size, n_actions, 
                 max_memory_size=100000, eps_min=0.01, eps_dec=5e-4):
        
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_memory_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.mem_counter = 0

        self.action_space = [i for i in range(n_actions)]
        self.Q_eval = DeepQNetwork(lr, input_dims, 256, 256, n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, new_state, done):
        # Find index of unoccupied space.
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done

        self.mem_counter += 1
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Take greedy action
            state = torch.tensor([observation]).to(self.Q_eval.device)
            # Get all action-value estimates from network
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            # Explore via random action
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        # If we haven't collected sufficient expereince to learn off of, return
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # Perform feed forward
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.discount_factor * torch.max(q_next, dim=1)[0]


        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec 
        else :
            self.eps_min