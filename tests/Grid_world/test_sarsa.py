import sys
import os
import numpy as np
import logging

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.grid_world import GridWorld
from algorithms.sarsa import Sarsa

logging.basicConfig(level=logging.INFO)


def test_sarsa():
    logging.info("Initializing environment...")
    env = GridWorld(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (1, 2), (2, 1), (3, 3)])
    logging.info("Initializing agent...")
    agent = Sarsa(env)
    logging.info("Starting training...")

    num_episodes = 10000  # Augmenté pour assurer une bonne convergence
    agent.train(num_episodes=num_episodes)

    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    logging.info("Politique obtenue:")
    logging.info(policy)
    logging.info("Fonction de valeur-action obtenue:")
    logging.info(action_value_function)

    logging.info("Policy Changes Over Episodes:")
    logging.info(agent.policy_changes)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL - Copie\tests\Grid_world\policy\sarsa_policy.npz')
    logging.info("Politique et fonctions sauvegardées dans 'sarsa_policy.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL - Copie\tests\Grid_world\policy\sarsa_policy.npz')
    loaded_policy = agent.get_policy()
    loaded_action_value_function = agent.get_action_value_function()
    logging.info("Politique chargée:")
    logging.info(loaded_policy)
    logging.info("Fonction de valeur-action chargée:")
    logging.info(loaded_action_value_function)

    # Démonstration de la politique
    logging.info("\nDémonstration de la politique:")
    state = env.reset()
    env.render()
    steps = 0
    max_steps = 100
    while state != env.goal and steps < max_steps:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        env.render()
        logging.info(f"Action: {action}, État: {state}, Récompense: {reward}")
        steps += 1
    if steps >= max_steps:
        logging.info("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    logging.info("\nInteraction manuelle:")
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = int(input("Entrez l'action (0 pour haut, 1 pour bas, 2 pour gauche, 3 pour droite): "))
        state, reward, done, _ = env.step(action)
        env.render()
        logging.info(f"État: {state}, Récompense: {reward}")


if __name__ == "__main__":
    test_sarsa()
