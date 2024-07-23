import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg pour éviter les problèmes liés à Qt
import numpy as np
import matplotlib.pyplot as plt

def analyze_results():
    data = np.load(r'D:\projet_DRL\tests\rock_paper_scissors\policy\algorithm_performance_rps.npz', allow_pickle=True)
    algorithms = data.files
    rewards = []
    durations = []

    for algo in algorithms:
        results = data[algo].item()
        rewards.append(results.get('total_reward'))
        durations.append(results.get('duration'))

    # Créer une figure avec deux sous-graphiques
    plt.figure(figsize=(12, 6))

    # Sous-graphe pour les récompenses totales
    plt.subplot(1, 2, 1)
    plt.barh(algorithms, rewards, color='skyblue')
    plt.xlabel('Total Reward')
    plt.ylabel('Algorithm')
    plt.title('Total Reward by Algorithm')

    # Sous-graphe pour les durées d'entraînement
    plt.subplot(1, 2, 2)
    plt.barh(algorithms, durations, color='salmon')
    plt.xlabel('Duration (s)')
    plt.ylabel('Algorithm')
    plt.title('Training Duration by Algorithm')

    # Ajuster l'agencement des sous-graphiques
    plt.tight_layout()

    # Sauvegarder la figure
    plt.savefig('algorithm_performance_comparison_rps.png')

    # Afficher la figure
    plt.show()

if __name__ == "__main__":
    analyze_results()
