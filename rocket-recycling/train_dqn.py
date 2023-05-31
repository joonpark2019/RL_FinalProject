import numpy as np
import torch
from rocket import Rocket
from policy_dqn import DQN
from policy_dqn import DQNAgent
import matplotlib.pyplot as plt
import utils
import os
import glob
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # this line was added to avoid kernel error within my environment

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'

    max_m_episode = 200000
    max_steps = 800
    BATCH_SIZE = 64
    decay = 0.995
    epsilon_init = 0.99
    epsilon_min = 0.1

    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []
    replay_memory = []
    epsilon = epsilon_init    
    
    net = DQNAgent(input_dim=env.state_dims, output_dim=env.action_dims, device = device)
    
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        state = env.reset()
        episode_memory = []
        for step_id in range(max_steps):
            action = net.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_memory.append((state, action, reward, next_state, done))
            state = next_state
            
            if episode_id % 10000 == 1:
                env.render()

            if done or step_id == max_steps-1:
                replay_memory.extend(episode_memory)
                if len(replay_memory) > BATCH_SIZE:
                    batch = random.sample(replay_memory, BATCH_SIZE)
                    net.update(batch, gamma=0.999)
                break
                
        epsilon = max(epsilon * decay, epsilon_min)

        REWARDS.append(np.sum([reward[2] for reward in episode_memory]))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum([reward[2] for reward in episode_memory])))

        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



