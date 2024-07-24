import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def extract_reward_and_duration(data):
    if isinstance(data, list):
        data = data[-1]  # Take the last iteration if it's a list

    total_reward = 0
    duration = 0

    if 'Q' in data:
        # Estimate total reward as sum of max Q-values
        Q = data['Q']
        if isinstance(Q, dict):
            total_reward = sum(max(state_values.values()) for state_values in Q.values())
        elif isinstance(Q, np.ndarray):
            total_reward = np.sum(np.max(Q, axis=1))

    if 'duration' in data:
        duration = data['duration']
    elif 'training_time' in data:
        duration = data['training_time']

    return total_reward, duration


def visualize_results(results):
    algorithms = list(results.keys())
    rewards = [result['reward'] for result in results.values()]
    durations = [result['duration'] for result in results.values()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Total Reward plot
    ax1.bar(algorithms, rewards)
    ax1.set_title('Total Reward per Algorithm')
    ax1.set_ylabel('Total Reward')
    ax1.tick_params(axis='x', rotation=45)

    # Training Duration plot
    ax2.bar(algorithms, durations)
    ax2.set_title('Training Duration per Algorithm')
    ax2.set_ylabel('Duration (seconds)')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Efficiency plot (Reward per unit time)
    efficiency = [r / d if d > 0 else 0 for r, d in zip(rewards, durations)]
    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, efficiency)
    plt.title('Algorithm Efficiency (Reward per second)')
    plt.ylabel('Reward/second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    results_folder = r"D:\projet_DRL - Copie\tests\test2\results"
    results = {}

    for filename in os.listdir(results_folder):
        if filename.endswith('.pkl'):
            file_path = os.path.join(results_folder, filename)
            data = load_pkl(file_path)

            algo_name = filename.split('_')[1]  # Assuming format is "EnvName_AlgoName_*.pkl"

            total_reward, duration = extract_reward_and_duration(data)

            results[algo_name] = {
                'reward': total_reward,
                'duration': duration
            }

            print(f"{algo_name}: Total Reward = {total_reward}, Duration = {duration} seconds")

    visualize_results(results)


if __name__ == "__main__":
    main()