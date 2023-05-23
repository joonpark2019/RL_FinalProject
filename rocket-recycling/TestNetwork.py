import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


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


class NetworkBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NetworkBase, self).__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        self.fcV1 = nn.Linear(in_features=self.mapping.output_dim, out_features = 128, bias=True)
        self.fcV2 = nn.Linear(in_features=128, out_features = 128, bias = True)
        self.fcV3 = nn.Linear(in_features=128, out_features = 128, bias = True)

        self.fcV4 = nn.Linear(in_features=128, out_features =output_dim, bias = True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.fcV1(x))
        x = self.relu(self.fcV2(x))
        x = self.relu(self.fcV3(x))
        x = self.fcV4(x)
        return x



class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.actor = NetworkBase(input_dim, output_dim)
        self.critic = NetworkBase(input_dim, 1)
        self.actor_target = NetworkBase(input_dim, output_dim)

        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=-1)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=5e-5)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=5e-5)


    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value
    
    def get_action(self, state, deterministic=False, exploration=0.01):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    
    @staticmethod
    def update_ac(network, actions, states, rewards, log_probs, values, masks, Qval, gamma=0.99):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        # print('log probs:', log_probs)
        # print('values:', values)

        action_tuple = torch.stack(actions)
        state_tuple = torch.stack(states)
        
        print('action_tuple', action_tuple)
        print('state_tuple', state_tuple)
        

        # advantage = Qvals - values
        # actor_loss = (-log_probs * advantage.detach()).mean()
        # critic_loss = 0.5 * advantage.pow(2).mean()
        # ac_loss = actor_loss + critic_loss

        # network.optimizer.zero_grad()
        # ac_loss.backward()
        # network.optimizer.step()

        # with torch.no_grad():
        #     delta = R + gamma*network.actor(S_next)-network.actor(S)
        #     GAE = gamma * lmbda * GAE + delta             
        #     G = gamma * G + R
        advantage = Qvals - values
        critic_loss = 0.5 * advantage.pow(2).mean()
            
        eps_clip = 0.1
        ratio = network.actor(state_tuple)[action_tuple]/network.actor_target(state_tuple)[action_tuple]
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage 
        actor_loss = actor_loss - torch.min(surr1, surr2)
        actor_loss = actor_loss.mean()


        network.critic_optimizer.zero_grad()
        critic_loss.step()
        network.critic_optimizer.step()

        network.actor_optimizer.zero_grad()
        actor_loss.step()
        network.actor_optimizer.step()



    