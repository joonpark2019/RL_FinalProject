import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython.display import clear_output
from rocket import Rocket
import os
import glob
import utils
import matplotlib.pyplot as plt

MAX_EPISODE = 20000
GAMMA = 0.99
MAX_STEP = 800
clip_epsilon = 0.2  # Clip parameter for PPO

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = self.Critic(state_dim)
        self.actor = self.Actor(state_dim, action_dim)

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim = None):
            super().__init__()
            self.critic_fc1 = nn.Linear(state_dim, 64)
            self.critic_fc2 = nn.Linear(64, 64)
            self.critic = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.critic_fc1(x))
            x = F.relu(self.critic_fc2(x))

            critic_output = self.critic(x)

            return critic_output
        
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.actor_fc1 = nn.Linear(state_dim, 64)
            self.actor_fc2 = nn.Linear(64, 64)
            self.actor = nn.Linear(64, action_dim)

        def forward(self, x):
            x = F.relu(self.actor_fc1(x))
            x = F.relu(self.actor_fc2(x))

            actor_output = F.softmax(self.actor(x), dim=-1)

            return actor_output


def ppo_train(env, actor_critic, num_episodes = MAX_EPISODE):
    return_history =[]
    
    for episode in range(num_episodes):
        # Generate an episode
        states, actions, rewards = generate_episode(env, actor_critic)
        G = 0
        returns =  []

    # Then for each step, we store the rewards to a variable R and states to S, and we calculate
     # returns as a sum of rewards
        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            A = actions[t]
            G = GAMMA * G + R
            returns.insert(0, G)

        # plot return for each episode

        # Value function update
        value_loss = fit_value_function(actor_critic, states, returns)
        print(f"Episode: {episode+1}, Value Loss: {value_loss.item()}")

        current_values = actor_critic.critic(torch.tensor(states, dtype=torch.float32).to(device))  # torch.Size([states_length, 9]), torch.Size([states_length, 1])
        current_values = current_values.squeeze(1)                                                  # torch.Size([states_length])
        
        # Compute TD-style advantage estimates based on the current value function
        advantages = compute_advantages(rewards, states, actor_critic)

        # Update the policy using Proximal Policy Optimization
        ppo_update(actor_critic, states, actions, advantages)

        return_history.append(G)
        if episode % 1 == 0:
            clear_output(wait=True)
            print('episode: {}, return: {:.1f}'.format(episode, return_history[-1]))   

        # save checkpoint
        if episode % 100 == 1:
            plt.figure()
            plt.plot(return_history[-1]), plt.plot(utils.moving_avg(return_history, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode,
                            'REWARDS': REWARDS,
                            'model_G_state_dict': actor_critic.state_dict()},
                        os.path.join(ckpt_folder, 'ckpt_' + str(episode).zfill(8) + '.pt'))

def generate_episode(env, actor_critic, max_step = MAX_STEP):
    # we initialize the list for storing states, actions, and rewards
    states, actions, rewards = [], [], []

    # Initialize the gym environment
    state = env.reset()
    terminated = False

    # generate episode for m_episode
    for step in range(max_step):
        states.append(state)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = actor_critic.actor(state_tensor)
        
        action = action_probs.multinomial(1).item()
        actions.append(action)

        next_state, reward, terminated, _ = env.step(action)
        rewards.append(reward)

        state = next_state

        if terminated:
            break
    

    return states, actions, rewards

def fit_value_function(actor_critic, states, returns):
    critic_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=0.001)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)       # [states_length, 8]
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)     # [states_length]
    
    values = actor_critic.critic(states_tensor)                                    # torch.Size([states_length, 9]), torch.Size([states_length, 1])
    values = values.squeeze(1)                                                 # torch.Size([states_length])

    value_loss = F.mse_loss(returns_tensor, values)

    critic_optimizer.zero_grad()
    value_loss.backward()
    critic_optimizer.step()

    return value_loss

def compute_advantages(rewards, states, actor_critic):
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    values = actor_critic.critic(states_tensor)
    values = values.squeeze(1).detach().cpu().numpy()

    advantages = []
    advantage = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_value - values[t]
        advantage = delta + GAMMA * advantage
        advantages.insert(0, advantage)
        next_value = values[t]

    return advantages

def ppo_update(actor_critic, states, actions, advantages):
    actor_optimizer = optim.Adam(actor_critic.actor.parameters(), lr=0.001)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

    # Compute old action probabilities
    action_probs_old = actor_critic.actor(states_tensor)
    old_log_probs = torch.log(action_probs_old.gather(1, actions_tensor.unsqueeze(1))).clone()

    # Compute new action probabilities
    action_probs = actor_critic.actor(states_tensor)
    log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1))).clone()

    # Compute surrogate objective function
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages_tensor
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor
    actor_loss = -torch.min(surr1, surr2).mean()

    # Update the policy using the actor_optimizer
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
        
if __name__ == '__main__':
    
    task = 'hover'  # 'hover' or 'landing'
    version = 1

    env = Rocket(task=task, max_steps=MAX_STEP)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    actor_critic = ActorCritic(state_dim=env.state_dims, action_dim=env.action_dims).to(device)

    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        actor_critic.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']


    ppo_train(env, actor_critic, num_episodes=MAX_EPISODE)


