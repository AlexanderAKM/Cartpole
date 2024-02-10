import argparse
from agents import dqn_agent, sarsa_agent, qlearning_agent
from agents.sarsa_agent import run as run_sarsa
from agents.dqn_agent import run as run_dqn
from agents.qlearning_agent import run as run_qlearning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RL Agents')
    parser.add_argument('--agent', type=str, choices=['dqn', 'sarsa', 'qlearning'], required=True, help='The type of agent to run')
    parser.add_argument('--episodes', type=int, default=1000, help='The number of episodes to run')
    
    args = parser.parse_args()

    if args.agent == 'dqn':
        dqn_agent.run_dqn(args.episodes)
    elif args.agent == 'sarsa':
        sarsa_agent.run_sarsa(args.episodes)
    elif args.agent == 'qlearning':
        qlearning_agent.run_qlearning(args.episodes)
