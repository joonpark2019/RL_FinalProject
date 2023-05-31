import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
class DQN(nn.Module):
    """
    Implementation of QNetwork
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = MLP(input_dim=input_dim, output_dim=output_dim)
        
    def forward(self, x):
        return self.network(x)
    
class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.dqn = DQN(input_dim, output_dim).to(device)
        self.target_dqn = DQN(input_dim, output_dim).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.00005)
        self.replay_buffer = ReplayBuffer(1000000)
    
    def get_action(self, state, exploration):
        
        if random.random() < exploration:
            return random.randint(0, self.output_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.dqn(state)
            return q_values.argmax().item()
            
    def update(self, batch, gamma=0.99):
        BATCH_SIZE = 64
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        current_q = self.dqn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_state).max(1)[0]
        expected_q = reward + gamma * next_q_values * (1 - done)

        loss = F.mse_loss(current_q, expected_q)

        # gradient initialization and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())