import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

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

    base_name = os.path.basename(csv_file_path)
    output_file_name = f"{base_name[:-4]}.png"

    output_path = os.path.join('data/output', output_file_name)
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot rewards from CSV file.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to CSV file containing rewards.')

    args = parser.parse_args()
    plot_rewards(args.file)


