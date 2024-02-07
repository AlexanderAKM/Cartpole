import numpy as np
import gymnasium as gym
import time
import math
import csv
import pandas as pd

env = gym.make("CartPole-v1")
#print(env.action_space.n)
#print(env.observation_space)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 6000
SHOW_EVERY = 2000


OBSERVATION = [20, 20, 20, 20]

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

q_table = np.random.uniform(low=0, high=1, size=(OBSERVATION + [env.action_space.n]))
#q_table.shape
#print(q_table)

def get_discrete_state(state, bins=OBSERVATION):
    #print(f"state: {state} and velocity: {velocity} and angular_velocity: {angular_velocity}")
    cart_position = np.digitize(state[0], np.linspace(-4.8, 4.8, bins[0])) - 1
    cart_velocity = np.digitize(state[1], np.linspace(-1, 1, bins[1])) - 1
    pole_angle = np.digitize(state[2], np.linspace(-0.418, 0.418, bins[2])) - 1
    pole_angular_velocity = np.digitize(state[3], np.linspace(-1, 1, bins[3])) - 1
    
    return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
 
rewards_episodes = {'Episode': [], 'Reward': []}

for episode in range(EPISODES + 1):
    state = env.reset()
    total_reward = 0
    discrete_state = get_discrete_state(state[0])
    terminated, truncated = False, False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * episode)

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not terminated and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)  # Unpack the tuple correctly
        new_discrete_state = get_discrete_state(new_state)  # Pass the new state array directly
        total_reward += reward
        #print(env.step(action))
        if render:
            env.render()
            time.sleep(0.01)

        
        if not terminated and not truncated:
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[new_discrete_state]) - current_q)
            q_table[discrete_state + (action,)] = new_q
        
        discrete_state = new_discrete_state
    
    
    rewards_episodes['Episode'].append(episode)
    rewards_episodes['Reward'].append(total_reward)
        
rewards_file_path = 'data/rewards_qlearning_6000.csv'
df_rewards = pd.DataFrame(rewards_episodes)
df_rewards.to_csv(rewards_file_path, index=False)

env.close()