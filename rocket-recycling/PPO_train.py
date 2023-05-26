
import numpy as np
import torch
from rocket import Rocket
# from TestNetwork import ActorCritic
from PPO_network import VNetwork, PolicyNetwork
import matplotlib.pyplot as plt
import utils
import os
import glob
import time
import datetime
from collections import deque

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gen_episode(environment):
    states = []
    actions = []
    rewards = []
    ratios = []
    state = environment.reset() 
    terminated = False

    while True:
        probs_target = pi_target(torch.FloatTensor(state))
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

if __name__ == '__main__':
    
    task = 'hover'  # 'hover' or 'landing'
    version = 2

    max_m_episode = 200000
    max_steps = 800
     #network and optimizer

    #hyperparameters:
    alpha = 0.001
    gamma = 0.99
    lmbda         = 0.99
    eps_clip      = 0.1
    K_epoch       = 4

    #create networks:
    pi = PolicyNetwork()
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)
    pi_target = PolicyNetwork()

    V = VNetwork()
    V_optimizer = torch.optim.Adam(V.parameters(), lr=alpha)  

    V = V.to(device)
    pi = pi.to(device)
    
    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)


    # if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
    #         # load the last ckpt
    #     checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
    #     #modify later:
    #     net.load_state_dict(checkpoint['model_G_state_dict'])
    #     last_episode_id = checkpoint['episode_id']
    #     REWARDS = checkpoint['REWARDS']


    episode = 0
    MAX_EPISODES = 20000
    reward_history =[]
    reward_history_100 = deque(maxlen=100)

    while episode < MAX_EPISODES:  # episode loop
        
        pi_target.load_state_dict(pi.state_dict())
        states, actions, rewards = gen_episode(env)
            
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
                
                ratio = pi(S)[A]/pi_target(S)[A]
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
    
        reward_history.append(np.sum(rewards))
        # reward_history_100.append(G)
        # avg = sum(reward_history_100) / len(reward_history_100)

        if episode % 10 == 1:
            print('episode id: %d, episode reward: %.3f'
                % (episode, np.sum(rewards)))
            plt.figure()
            plt.plot(reward_history), plt.plot(utils.moving_avg(reward_history, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(version).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode,
                            'REWARDS': reward_history,
                            'model_pi_state_dict': pi.state_dict(),
                            'model_V_state_dict': V.state_dict()},
                           os.path.join(ckpt_folder, 'ckpt_' + str(version).zfill(8) + '.pt'))