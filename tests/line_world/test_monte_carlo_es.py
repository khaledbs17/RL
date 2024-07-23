import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.line_world import LineWorld
from algorithms.monte_carlo_es import MonteCarloES

def test_monte_carlo_es():
    print("Initializing environment...")
    env = LineWorld(length=10, start=0, goal=9)
    print("Initializing agent...")
    agent = MonteCarloES(env)
    print("Starting training...")

    num_episodes = 10000  # Réduire le nombre d'épisodes pour les tests
    for episode in range(num_episodes):
        agent.train(num_episodes=1)
        print(f"Training progress: Episode {episode + 1}/{num_episodes}")

    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL - Copie\tests\line_world\policy\monte_carlo_es_policy.npz')
    print("Politique et fonctions sauvegardées dans 'monte_carlo_es_policy.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL - Copie\tests\line_world\policy\monte_carlo_es_policy.npz')
    loaded_policy = agent.get_policy()
    loaded_action_value_function = agent.get_action_value_function()
    print("Politique chargée:")
    print(loaded_policy)
    print("Fonction de valeur-action chargée:")
    print(loaded_action_value_function)

    # Démonstration de la politique
    print("\nDémonstration de la politique:")
    state = env.reset()
    env.render()
    steps = 0
    max_steps = 100
    while state != env.goal and steps < max_steps:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, État: {state}, Récompense: {reward}")
        steps += 1
    if steps >= max_steps:
        print("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = int(input("Entrez l'action (0 pour gauche, 1 pour droite): "))
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"État: {state}, Récompense: {reward}")

if __name__ == "__main__":
    test_monte_carlo_es()
