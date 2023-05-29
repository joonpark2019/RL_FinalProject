import random
import numpy as np
import torch
import utils
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
import gymnasium as gym

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
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)


    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        actor_output = F.softmax(self.actor(x), dim=-1)     # probability
        critic_output = self.critic(x)                      # value
        
        return actor_output, critic_output
        
    def get_action(self, state):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        action_id = torch.multinomial(probs, 1).item()
        # log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, probs, value

    def target_pred(self, state):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        y = self.actor_target(state)
        probs = self.softmax(y)
        probs = probs[0, :]

        action_id = torch.multinomial(probs, 1).item()
        
        # if deterministic:
        #     action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        # else:
        #     if random.random() < exploration:  # exploration
        #         action_id = random.randint(0, self.output_dim - 1)
        #     else:
        #         action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))
        # log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, probs
    
def ppo_update(actor_critic, states, actions, log_probs, returns, advantages, clip_epsilon):
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    
    step = 10
    for _ in range(step):
        actor_output, critic_output = actor_critic(states)
        new_log_probs = torch.log(actor_output.gather(1, actions.unsqueeze(1)).squeeze(1))
        ratio = torch.exp(new_log_probs - log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        critic_loss = F.mse_loss(critic_output.squeeze(1), returns)

        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def ppo_train(env, actor_critic, num_epochs, num_steps, clip_epsilon, gamma, lam, ckpt_folder):
    REWARDS = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        for _ in range(num_steps):
            state = env.reset()
            total_reward = 0
            for _ in range(1000):  # Maximum episode length
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, value = actor_critic(state)

                action = action_probs.multinomial(1).detach()   #! 뭐지?
                log_prob = torch.log(action_probs.gather(1, action))

                next_state, reward, done, _ = env.step(action.item())
                log_probs.append(log_prob)
                values.append(value)
                states.append(state)  # Append the current state to the states list
                actions.append(action.item())
                rewards.append(reward)
                masks.append(1 - done)            
                
                state = next_state
                total_reward += reward

                if done:
                    break

        # Compute returns and advantages
        returns = compute_returns(rewards, masks, gamma)    
        advantages = compute_advantages(rewards, values, masks, gamma, lam)

        # Convert lists to tensors
        old_log_probs = torch.cat(log_probs).detach().to(device)
        old_values = torch.cat(values).detach().to(device)
        states = torch.cat(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.cat(returns).to(device)
        advantages = torch.cat(advantages).to(device)


        # Perform PPO update
        ppo_update(actor_critic, states, actions, old_log_probs, returns, advantages, clip_epsilon)

        print(f"Epoch: {epoch+1}, Total Reward: {total_reward}")
        
        REWARDS.append(total_reward)

        import matplotlib.pyplot as plt
        import os

        if epoch % 100 == 1:
            print('episode id: %d, episode reward: %.3f'
                % (epoch, total_reward))
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(epoch).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': epoch,
                            'REWARDS': REWARDS,
                            'model_G_state_dict': actor_critic.state_dict()},
                        os.path.join(ckpt_folder, 'ckpt_' + str(epoch).zfill(8) + '.pt'))


def compute_returns(rewards, masks, gamma):
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(torch.tensor([0]), torch.tensor([R]))
    return returns

def compute_advantages(rewards, values, masks, gamma, lam):
    deltas = []
    advantages = []
    prev_value = 0
    prev_advantage = 0
    for step in reversed(range(len(rewards)-1)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        deltas.insert(0, delta)
        advantage = delta + gamma * lam * prev_advantage * masks[step]
        advantages.insert(0, advantage)
        prev_value = values[step]
        prev_advantage = advantage
    return advantages
