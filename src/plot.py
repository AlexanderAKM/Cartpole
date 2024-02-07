import matplotlib.pyplot as plt
import pandas as pd

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
    plt.savefig('data/SARSA_20000.png')
    plt.show()

plot_rewards('data/rewards_SARSA_20000.csv')

