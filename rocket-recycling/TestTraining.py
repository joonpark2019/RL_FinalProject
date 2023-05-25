

import numpy as np
import torch
from rocket import Rocket
from TestNetwork import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob
import time
import datetime

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    task = 'hover'  # 'hover' or 'landing'
    version = 1

    max_m_episode = 200000
    max_steps = 800

    
    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
            # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']


    for episode_id in range(last_episode_id, max_m_episode):

            # training loop
        rewards, log_probs, values, masks = [], [], [], []
            #actor_critic implementation:
        net.actor_target.load_state_dict(net.actor.state_dict())

        rewards, probs, probs_target, values, masks = [], [], [], [], []
        states = []
        actions = []
        rewards = []
        terminated = False

        state = env.reset() 
        terminated = False
        for step_id in range(max_steps):
            
            action, prob_vect, value = net.get_action(state)

            state, reward, terminated, _ = env.step(action) 

            _, prob_target_vect = net.target_pred(state)

            # print("prob_vect:", prob_vect)
            # print("prob_vect_target:", prob_target_vect)


            # rewards.append(reward)
            probs.append(prob_vect)
            probs_target.append(prob_target_vect)
            values.append(value)
            masks.append(1-terminated)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            # print("###states####:")
            # print(states)

            # print("###actions####:")
            # print(actions)

            # print("###rewards####:")
            # print(rewards)

            

            if terminated or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(actions, states, rewards, probs, probs_target, values, masks, Qval, gamma=0.999)
                break
        REWARDS.append(np.sum(rewards))
        # print('episode id: %d, episode reward: %.3f'
        #         % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 1:
            print('episode id: %d, episode reward: %.3f'
                % (episode_id, np.sum(rewards)))
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(version).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                            'REWARDS': REWARDS,
                            'model_G_state_dict': net.state_dict()},
                           os.path.join(ckpt_folder, 'ckpt_' + str(version).zfill(8) + '.pt'))
