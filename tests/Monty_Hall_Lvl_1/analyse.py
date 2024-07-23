import os
import numpy as np
import matplotlib.pyplot as plt


def analyze_results():
    # Définir les noms des fichiers de politiques pour chaque algorithme
    filenames = {
        'Dyna-Q': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\dyna_q_policy_MH1.npz',
        'Q-Learning': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\q_learning_policy_MH1.npz',
        'SARSA': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\sarsa_policy_MH1.npz',
        'Monte Carlo ES': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\monte_carlo_es_policy_MH1.npz',
        'Off-Policy MCC': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\off_policy_mcc_MH1.npz',
        'On-Policy First Visit MCC': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\on_policy_first_visit_MH1.npz',
        'Policy Iteration': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\policy_iteration_MH1.npz',
        'Value Iteration': r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\value_iteration_policy_MH1.npz'
    }

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
    analyze_results()
