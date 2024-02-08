# CartPole Reinforcement Learning Solutions

This repository features implementations of three reinforcement learning (RL) algorithms applied to the classic CartPole task from OpenAI Gym: Q-learning, SARSA, and Deep Q-Network (DQN). Each algorithm demonstrates a unique approach to learning how to balance a pole on a moving cart.

## Overview

The CartPole task is a benchmark challenge in reinforcement learning. The agent controls a cart with a pole attached; the goal is to prevent the pole from falling over. The project showcases:

- **Q-learning:** An off-policy algorithm for learning the optimal action-value function.
- **SARSA:** An on-policy algorithm that updates Q-values using the action performed by the current policy.
- **Deep Q-Network (DQN):** Uses deep neural networks to approximate the Q-value function, with techniques like experience replay and fixed Q-targets for stability.

## Getting Started

### Installation

First, clone this repository:
https://github.com/AlexanderAKM/Cartpole.git

Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* OpenAI Gym for providing the CartPole environment.
* Documentation of Pytorch on DQN implementation for cartpole: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
