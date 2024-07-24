import sys
import os
import numpy as np
import logging

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from algorithms.monte_carlo_es import MonteCarloES
from environments.secret_envs_wrapper import SecretEnv1

logging.basicConfig(level=logging.INFO)

def test_monte_carlo_es():
    logging.info("Initializing environment...")
    env = SecretEnv1()
    logging.info("Initializing agent...")
    agent = MonteCarloES(env)
    logging.info("Starting training...")
    num_episodes = 10000  # Vous pouvez ajuster ce nombre pour des tests plus rapides

    for episode in range(num_episodes):
        logging.info(f"Starting episode {episode + 1}/{num_episodes}...")
        episode_data = agent.generate_episode()
        agent.update_policy(episode_data)
        if episode % 1000 == 0:
            logging.info(f"Episode {episode + 1}/{num_episodes} completed.")

    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    logging.info("Politique obtenue:")
    logging.info(policy)
    logging.info("Fonction de valeur-action obtenue:")
    logging.info(action_value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save('monte_carlo_es_policy_secret_1.npz')
    logging.info("Politique et fonctions sauvegardées dans 'monte_carlo_es_policy_secret_1.npz'.")

    # Chargement de la politique et des fonctions
    agent.load('monte_carlo_es_policy_secret_1.npz')
    loaded_policy = agent.get_policy()
    loaded_action_value_function = agent.get_action_value_function()

    logging.info("Politique chargée:")
    logging.info(loaded_policy)
    logging.info("Fonction de valeur-action chargée:")
    logging.info(loaded_action_value_function)

    # Démonstration de la politique
    logging.info("\nDémonstration de la politique:")
    env.reset()
    env.display()
    state = int(env.state_id())
    while not env.is_game_over():
        action = int(loaded_policy[state])
        if env.is_forbidden(action):
            logging.info(f"Forbidden action: {action}, choosing another action")
            action = agent.select_action(state)
        env.step(action)
        env.display()
        state = int(env.state_id())
        logging.info(f"Action: {action}, État: {state}, Récompense: {env.score()}")

if __name__ == "__main__":
    test_monte_carlo_es()
