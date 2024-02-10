'''SARSA algorithm implemented by following the book, "Reinforcement Learning: An Introduction". '''

import numpy as np
import gymnasium as gym
import time
import math
import pandas as pd

def run(episodes):
    env = gym.make("CartPole-v1")

    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    SHOW_EVERY = 2000


    OBSERVATION = [20, 20, 20, 20]

    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001

    q_table = np.random.uniform(low=0, high=1, size=(OBSERVATION + [env.action_space.n]))

    def get_discrete_state(state, bins=OBSERVATION):
        cart_position = np.digitize(state[0], np.linspace(-4.8, 4.8, bins[0])) - 1
        cart_velocity = np.digitize(state[1], np.linspace(-1, 1, bins[1])) - 1
        pole_angle = np.digitize(state[2], np.linspace(-0.418, 0.418, bins[2])) - 1
        pole_angular_velocity = np.digitize(state[3], np.linspace(-1, 1, bins[3])) - 1
        
        return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
    
    rewards_episodes = {'Episode': [], 'Reward': []}

    def get_action(epsilon, discrete_state):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n) 

        return action   

    for episode in range(episodes + 1):
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

        action = get_action(epsilon, discrete_state)

        while not terminated and not truncated:
            new_state, reward, terminated, truncated, _ = env.step(action)  # Unpack the tuple correctly
            total_reward += reward
            
            new_discrete_state = get_discrete_state(new_state)
            new_action = get_action(epsilon, new_discrete_state)
            
            if render:
                env.render()
                time.sleep(0.01)
            
            if not terminated and not truncated:
                current_q = q_table[discrete_state + (action,)]
                new_q = q_table[new_discrete_state + (new_action,)]
                new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * new_q - current_q)
                q_table[discrete_state + (action,)] = new_q
            
            discrete_state = new_discrete_state
            action = new_action
        
        rewards_episodes['Episode'].append(episode)
        rewards_episodes['Reward'].append(total_reward)
            
    rewards_file_path = f'data/input/rewards_sarsa_{episodes}.csv'
    df_rewards = pd.DataFrame(rewards_episodes)
    df_rewards.to_csv(rewards_file_path, index=False)

    env.close()


#if __name__ == "__main__":
    #run(episodes=20000)