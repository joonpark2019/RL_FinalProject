import numpy as np
import torch
from rocket import Rocket
# from TestNetwork import ActorCritic
# from PPO_network import VNetwork, PolicyNetwork
import matplotlib.pyplot as plt
import utils
import os
import glob
import time
import datetime
from collections import deque
import torch.optim as optim

import torch.nn as nn

from typing import Any

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
    help="resume with existing checkpoint")

settings = parser.parse_args()

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

# Based on the code from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# Network architecture and hyperparameters are based on : https://arxiv.org/pdf/2006.05990.pdf
# The code below is taken from: https://github.com/huggingface/deep-rl-class/blob/main/notebooks/unit8/unit8_part1.ipynb


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class VNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.mapping.output_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.critic(x)
        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.mapping.output_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_dim), std=1.0),
        )
        
    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.actor(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x


def gen_episode(environment, policy_target, device, max_step = 800):
    states = []
    actions = []
    rewards = []
    ratios = []
    state = environment.reset() 
    terminated = False

    for step in range(max_step):
        probs_target = policy_target(torch.FloatTensor(state).to(device))
        action = torch.multinomial(probs_target, 1).item()
        
        next_state, reward, terminated, _ = environment.step(action) 
        #must add:
#         env.render()
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if terminated:
            break  
        
        state = next_state
    return states, actions, rewards




task = 'hover'  # 'hover' or 'landing'
version = 5

max_m_episode = 200000
max_steps = 800
     #network and optimizer

    #hyperparameters:
alpha = 2.5e-4
gamma = 0.99
lmbda         = 0.99
eps_clip      = 0.1
K_epoch       = 4

env = Rocket(task=task, max_steps=max_steps)


#create networks:
pi = PolicyNetwork(env.state_dims, env.action_dims)
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)
pi_target = PolicyNetwork(env.state_dims, env.action_dims)

V = VNetwork(env.state_dims)
V_optimizer = torch.optim.Adam(V.parameters(), lr=alpha)  

V = V.to(device)
pi = pi.to(device)
pi_target = pi_target.to(device)

episode = 0
MAX_EPISODES = 20000
reward_history =[]
reward_history_100 = deque(maxlen=100)

# resume 시키려면 
if settings.resume:
    resume_episode_id = -1
    
    if settings.resume is not None and os.path.isfile(settings.resume) and os.path.exists(settings.resume):
        resume_ckpt = settings.resume
        state_dict = torch.load(resume_ckpt, map_location=torch.device(device))      # loaded model
        resume_episode_id = state_dict['episode_id']
        pi.load_state_dict(state_dict['model_pi_state_dict'])
        V.load_state_dict(state_dict['model_V_state_dict'])
        if state_dict['model_pi_optimizer'] is not None:
            pi_optimizer.load_state_dict(state_dict['model_pi_optimizer'])
        if state_dict['model_V_optimizer'] is not None:
            V_optimizer.load_state_dict(state_dict['model_V_optimizer'])

        print("Resuming training with checkpoint: {} from episode {}".format(resume_ckpt, resume_episode_id))

        # set parameters
        episode = 0 if resume_episode_id < 0 else (resume_episode_id + 1)
        reward_history = state_dict['REWARDS']
    
# 저장시킬 ckpt folder
ckpt_folder = os.path.join('./', task + '_ckpt')
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)


while episode < MAX_EPISODES:  # episode loop
    pi_target.load_state_dict(pi.state_dict())
    states, actions, rewards = gen_episode(env, pi_target, device, max_steps)
            
    episode += 1    
    for k in range(1,K_epoch):
        loss1 = 0
        loss2 = 0
        GAE = 0
        G = 0
        for t in range(len(states) - 2, -1, -1):
            S = states[t]
            A = actions[t]
            R = rewards[t]
            S_next = states[t+1]
                
            S=torch.FloatTensor(S).to(device)
            A=torch.tensor(A, dtype=torch.int8).to(device)
            S_next=torch.FloatTensor(S_next).to(device)
                
            with torch.no_grad():
                delta = R + gamma*V(S_next)-V(S)
                GAE = gamma * lmbda * GAE + delta           
                G = gamma * G + R


            actor_output = pi(S)
            actor_target_output = pi_target(S)
            actor_output = actor_output.view(-1).to(device)
            actor_target_output = actor_target_output.view(-1).to(device)

                
                
                # ratio = pi(S)[A]/pi_target(S)[A]
            ratio = actor_output[A] / actor_target_output[A]
                # print("ratio:", ratio)
            surr1 = ratio * (gamma**t)* GAE
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * (gamma**t)* GAE 
            loss1 = loss1 - torch.min(surr1, surr2)
            loss2 = loss2 + (G - V(S))**2
        loss2 = loss2/len(states)
                
        pi_optimizer.zero_grad()
        loss1.backward()
        pi_optimizer.step()
            
        V_optimizer.zero_grad()
        loss2.backward()
        V_optimizer.step() 
    
    reward_history.append(G)

    if episode % 10 == 1:
        print('episode id: %d, episode return: %.3f'
                % (episode, G))
        plt.figure()
        plt.plot(reward_history), plt.plot(utils.moving_avg(reward_history, N=50))
        plt.legend(['episode reward', 'moving avg'], loc=2)
        plt.xlabel('m episode')
        plt.ylabel('return')
        plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(version).zfill(8) + '.jpg'))
        plt.close()

        torch.save({'episode_id': episode,
                            'REWARDS': reward_history,
                            'model_pi_state_dict': pi.state_dict(),
                            'model_V_state_dict': V.state_dict(),
                            'model_pi_optimizer': pi_optimizer.state_dict(),
                            'model_V_optimizer': V_optimizer.state_dict()},
                           os.path.join(ckpt_folder, 'ckpt_' + str(version).zfill(8) + '.pt'))