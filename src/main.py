import argparse
from agents import dqn_agent, sarsa_agent, qlearning_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RL Agents')
    parser.add_argument('--agent', type=str, choices=['dqn', 'sarsa', 'qlearning'], required=True, help='The type of agent to run')

    args = parser.parse_args()

    if args.agent == 'dqn':
        dqn_agent.run()
    elif args.agent == 'sarsa':
        sarsa_agent.run()
    elif args.agent == 'qlearning':
        qlearning_agent.run()
        