import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(csv_file_path):
    data = pd.read_csv(csv_file_path)

    plt.figure(figsize=(10,5))
    plt.plot(data['Episode'], data['Reward'], label = 'Reward per Episode')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Rewards over Steps')
    plt.legend()
    plt.show()

plot_rewards('data/rewards.csv')

