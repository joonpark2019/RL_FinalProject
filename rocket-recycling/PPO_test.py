import torch
from rocket import Rocket
from PPO_network import VNetwork, PolicyNetwork
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800
    gamma = 0.99
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    print(ckpt_dir)
    env = Rocket(task=task, max_steps=max_steps)
    pi = PolicyNetwork(env.state_dims, env.action_dims)
    
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, map_location=torch.device(device))
        pi.load_state_dict(checkpoint['model_pi_state_dict'])

    state = env.reset()
    episode_returns = list()
    for i in range(100):
        _, _, rewards = gen_episode(env, pi, device, max_steps)
        G = 0
        for t in range(len(rewards) - 2, -1, -1):
                R = rewards[t]
                G = gamma * G + R

        episode_returns.append(G)


    average_return = sum(episode_returns) / len(episode_returns)
    print(average_return)
