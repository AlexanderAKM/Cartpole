import numpy as np
import gymnasium as gym
import time
import math

env = gym.make("CartPole-v1")
#print(env.action_space.n)
#print(env.observation_space)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 60000
total = 0
total_reward = 0
prior_reward = 0

Observation = [30, 30, 50, 50]
win_size = np.array([0.25, 0.25, 0.01, 0.1])

epsilon = 1
epsilon_decay = 0.99995

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape
print(q_table)

def get_discrete_state(state):
    discrete_state = state/ win_size+ np.array([15, 10, 1, 10])
    return tuple(discrete_state.ast)
                 

for episode in range(EPISODES + 1):
    

