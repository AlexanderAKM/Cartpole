import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plot_rewards(csv_file_path):
    rolling_window = 50
    data = pd.read_csv(csv_file_path)
    data['Reward_MA'] = data['Reward'].rolling(rolling_window).mean()

    plt.figure(figsize=(10,5))
    plt.plot(data['Episode'], data['Reward'], label = 'Reward per Episode', color = 'blue', alpha = 0.3)
    plt.plot(data['Episode'], data['Reward_MA'], label = f'{rolling_window}-Episode Moving Average', color = 'red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.savefig(f'{csv_file_path[0:-4]}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot rewards from CSV file.')
    parser.add_argument('-f', '--file', type=str, help='Path to CSV file containing rewards.')

    args = parser.parse_args()
    if args.file:
        plot_rewards(args.file)

#plot_rewards('data/rewards_DQN_600.csv')

