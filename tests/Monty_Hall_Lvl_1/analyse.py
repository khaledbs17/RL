import os
import numpy as np
import matplotlib.pyplot as plt


def analyze_policy_convergence(filenames, num_episodes):
    plt.figure(figsize=(12, 6))

    for algo, filepath in filenames.items():
        if os.path.exists(filepath):
            try:
                data = np.load(filepath, allow_pickle=True)
                policy_history = data['policy_history']  # Assuming this key holds policy data over episodes
                policy_changes = [np.sum(policy_history[i] != policy_history[i + 1]) for i in
                                  range(len(policy_history) - 1)]

                plt.plot(policy_changes, label=algo)

            except KeyError as e:
                print(f"KeyError: {e} in file {filepath}")
            except Exception as e:
                print(f"Unexpected error: {e} in file {filepath}")
        else:
            print(f"File {filepath} does not exist.")

    plt.xlabel('Episodes')
    plt.ylabel('Number of Policy Changes')
    plt.title('Policy Convergence Over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_results(filenames):
    # Définir les noms des fichiers de politiques pour chaque algorithme

    rewards = []
    durations = []
    algorithms = []

    for algo, filepath in filenames.items():
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                rewards.append(data['total_reward'].item())
                durations.append(data['duration'].item())
                algorithms.append(algo)
            except KeyError as e:
                print(f"KeyError: {e} in file {filepath}")
            except Exception as e:
                print(f"Unexpected error: {e} in file {filepath}")
        else:
            print(f"File {filepath} does not exist.")

    # Affichage des résultats
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(algorithms, rewards, color='skyblue')
    plt.xlabel('Total Reward')
    plt.title('Total Rewards per Algorithm')

    plt.subplot(1, 2, 2)
    plt.barh(algorithms, durations, color='lightgreen')
    plt.xlabel('Duration (seconds)')
    plt.title('Training Duration per Algorithm')

    plt.tight_layout()
    plt.savefig('algorithm_performance_comparison.png')
    plt.show()


if __name__ == "__main__":
    filenames = {
        'Dyna-Q': r'D:\projet_DRL - Copie\tests\Grid_world\policy\dyna_q_policy.npz',
        'Q-Learning': r'D:\projet_DRL - Copie\tests\Grid_world\policy\q_learning_policy.npz',
        'SARSA': r'D:\projet_DRL - Copie\tests\Grid_world\policy\sarsa_policy.npz',
        'Monte Carlo ES': r'D:\projet_DRL - Copie\tests\Grid_world\policy\monte_carlo_es_policy.npz',
        'Off-Policy MCC': r'D:\projet_DRL - Copie\tests\Grid_world\policy\off_policy_mcc_policy.npz',
        'On-Policy First Visit MCC': r'D:\projet_DRL - Copie\tests\Grid_world\policy\on_policy_first_visit_mcc_policy.npz',
        'Policy Iteration': r'D:\projet_DRL - Copie\tests\Grid_world\policy\policy_iteration_policy.npz',
        'Value Iteration': r'D:\projet_DRL - Copie\tests\Grid_world\policy\value_iteration_policy.npz'
    }

    analyze_results(filenames)
    analyze_policy_convergence(filenames, 1000)
