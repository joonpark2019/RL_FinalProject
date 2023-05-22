import numpy as np
# import matplotlib.pyplot as plt
# from IPython import display as ipythondisplay
from collections import deque
# import torch
import time
import datetime
import os

filepath = os.getcwd()

os.makedirs(f"{filepath}/CheckPoints/VNetwork/training", exist_ok=True)
os.makedirs(f"{filepath}/CheckPoints/VNetwork/validation", exist_ok=True)

os.makedirs(f"{filepath}/CheckPoints/PolicyNetwork/training", exist_ok=True)
os.makedirs(f"{filepath}/CheckPoints/PolicyNetwork/validation", exist_ok=True)

vnet_save_path = filepath + "/CheckPoints/VNetwork/training"
policy_save_path = filepath + "/CheckPoints/PolicyNetwork/training"

import cv2
from rocket import Rocket


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


task = 'hover'  # 'hover' or 'landing'
max_episode = 5
max_steps = 800


import random
import torch
from collections import deque

alpha = 0.001
gamma = 0.99
lmbda         = 0.99
eps_clip      = 0.1
K_epoch       = 4

env = Rocket(task=task, max_steps=max_steps)


# network and optimizer
pi = PolicyNetwork()
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)
pi_target = PolicyNetwork()

V = VNetwork()
V_optimizer = torch.optim.Adam(V.parameters(), lr=alpha)  

if torch.cuda.is_available():
  V = V.cuda()
  pi = pi.cuda()

best_avg_reward = 0
    
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def gen_episode():
    states = []
    actions = []
    rewards = []
    ratios = []
    state = env.reset() 
    terminated = False
    truncated = False
    while True:
        probs_target = pi_target(torch.FloatTensor(state))
        action = torch.multinomial(probs_target, 1).item()
        
        next_state, reward, terminated, _ = env.step(action) 
        #must add:
#         env.render()
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if terminated:
            break  
        
        state = next_state
    return states, actions, rewards


episode = 0
MAX_EPISODES = 20000
reward_history =[]
reward_history_100 = deque(maxlen=100)

while episode < MAX_EPISODES:  # episode loop
    
    pi_target.load_state_dict(pi.state_dict())
    states, actions, rewards = gen_episode()
        
        
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
            
            S=torch.FloatTensor(S)
            A=torch.tensor(A, dtype=torch.int8)
            S_next=torch.FloatTensor(S_next)
            
            with torch.no_grad():
                delta = R + gamma*V(S_next)-V(S)
                GAE = gamma * lmbda * GAE + delta             
                G = gamma * G + R
            
            ratio = pi(S)[A]/pi_target(S)[A]
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
    reward_history_100.append(G)
    avg = sum(reward_history_100) / len(reward_history_100)

    #save checkpoints intermittently
    


    episode = episode + 1
    if episode % 10 == 0:
        print('episode: {}, Return: {:.1f}, avg: {:.1f}'.format(episode, G, avg))
        if avg > best_avg_reward:
          best_vnet_state = V.state_dict()
          torch.save(best_vnet_state, vnet_save_path + '/' + date + '_best_v_state.pt')
          best_pi_state = pi_target.state_dict()
          torch.save(best_pi_state, policy_save_path + '/' + date + '_best_pi_state.pt')
    