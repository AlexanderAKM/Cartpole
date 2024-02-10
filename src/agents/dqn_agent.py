'''Mainly coded with help of Pytorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html'''
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

def run(episodes):

    env = gym.make("CartPole-v1")

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen = capacity)
        
        def push(self, *args):
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        
        def __len__(self):
            return len(self.memory)
        
    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)
        
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_actions = env.action_space.n

    state, info = env.reset()
    n_observations = len(state)

    '''The policy_net uses a NN to approximate the Q-value function. Input to this network is the state
    of the environment, and output are the predicted Q-values for each possible action in that state.'''
    policy_net = DQN(n_observations, n_actions).to(device)
    '''The target_net is a clone of the policy network, but with frozen weights. The weights are updated
    less frequently, which helps stabilize learning by providing a stable target for the Q-values updates.'''
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict()) # Ensures target's and policy's weights are same.

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        esp_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1 * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > esp_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]],
                                device = device, dtype=torch.long)

    episode_durations = []
    rewards_episodes = {'Episode': [], 'Reward': []}

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                                device = device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = \
            target_net(non_final_next_states).max(1).values
            
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if terminated or truncated:
                episode_durations.append(t + 1)
                break
        
        rewards_episodes['Episode'].append(episode)
        rewards_episodes['Reward'].append(total_reward)

    rewards_file_path = f'data/input/rewards_dqn_{episodes}.csv'
    df_rewards = pd.DataFrame(rewards_episodes)
    df_rewards.to_csv(rewards_file_path, index=False)

    env.close()