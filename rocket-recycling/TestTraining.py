

import numpy as np
import torch
from rocket import Rocket
from PPONet import ActorCritic, ppo_train
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

    max_m_episode = 5000
    max_steps = 800

    
    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    net = ActorCritic(state_dim=env.state_dims, action_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']


    ppo_train(env, net, num_epochs=800000, num_steps=200, clip_epsilon=0.2, gamma=0.99, lam=0.95, ckpt_folder=ckpt_folder)


