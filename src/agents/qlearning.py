import numpy as np
import gymnasium as gym
import time
import math

env = gym.make("CartPole-v1")
#print(env.action_space.n)
#print(env.observation_space)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 2000


Observation = [30, 30, 50, 50]
win_size = (env.observation_space.high - env.observation_space.low) / Observation

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape
print(q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / win_size
    return tuple(discrete_state.astype(np.int))
                 

for episode in range(EPISODES + 1):
    discrete_state = get_discrete_state(env.reset())
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * episode)

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        print(new_state)
        if render:
            env.render()
            time.sleep(0.01)

        
        if not done:
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[new_discrete_state]) - current_q)
            q_table[discrete_state + (action,)] = new_q
            
        