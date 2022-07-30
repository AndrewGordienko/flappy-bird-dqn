import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

LEARNING_RATE = 0.001
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
MEM_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.9988
EXPLORATION_MIN = 0.1

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = 5
        self.action_space = 2

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, 5),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, 5),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class Agent():
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX

        self.learn_step_counter = 0
        self.net_copy_interval = 10

        self.action_network = Network()
        self.target_network = Network()

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)

        state = torch.tensor(state).float()
        q_values = self.action_network(state)
        action = torch.argmax(q_values)

        return action
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.action_network(states)[batch_indices, actions] 
        next_q_values = self.target_network(states_)
        actions_ = self.action_network(states_).max(dim=1)[1]
        actions_ = next_q_values[batch_indices, actions_]

        q_target = rewards + GAMMA * actions_ * dones
        td = q_target - q_values

        self.action_network.optimizer.zero_grad()
        loss = ((td ** 2.0)).mean()
        loss.backward()
        self.action_network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_network.load_state_dict(self.action_network.state_dict())

        self.learn_step_counter += 1
    
    def transfer(self):
        self.target_network.load_state_dict(self.action_network.state_dict())
    
    

