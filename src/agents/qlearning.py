import numpy as np
import gymnasium as gym
import time
import math

env = gym.make("CartPole-v1", render_mode="rgb_array")
#print(env.action_space.n)
#print(env.observation_space)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 2000


OBSERVATION = [30, 30, 50, 50]

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

q_table = np.random.uniform(low=0, high=1, size=(OBSERVATION + [env.action_space.n]))
#q_table.shape
#print(q_table)

def get_discrete_state(state, bins=OBSERVATION):
    print(f"State received: {state}")
    velocity = np.linspace(-1, 1, bins[1])
    angular_velocity = np.linspace(-1, 1, bins[3])
    #print(f"state: {state} and velocity: {velocity} and angular_velocity: {angular_velocity}")
    cart_position = np.digitize(state[0], np.linspace(-4.8, 4.8, bins[0]))
    cart_velocity = np.digitize(state[1], velocity)
    pole_angle = np.digitize(state[2], np.linspace(-0.418, 0.418, bins[2]))
    pole_angular_velocity = np.digitize(state[3], angular_velocity)
    
    return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
                 

for episode in range(EPISODES + 1):
    state = env.reset()
    print(f"state: {state}")
    discrete_state = get_discrete_state(state[0])
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

        new_state, reward, done, _, _ = env.step(action)  # Unpack the tuple correctly
        new_discrete_state = get_discrete_state(new_state)  # Pass the new state array directly
        
        if render:
            env.render()
            time.sleep(0.01)

        
        if not done:
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[new_discrete_state]) - current_q)
            q_table[discrete_state + (action,)] = new_q
        
        discrete_state = new_discrete_state

env.close()