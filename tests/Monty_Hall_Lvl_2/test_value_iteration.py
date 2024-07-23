import sys
import os
import logging

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.monty_hall_level_2 import MontyHallLevel2
from algorithms.value_iteration import ValueIteration

logging.basicConfig(level=logging.INFO)

def test_value_iteration():
    logging.info("Initializing environment...")
    env = MontyHallLevel2()
    logging.info("Initializing agent...")
    agent = ValueIteration(env)
    logging.info("Starting training...")
    agent.train()
    policy = agent.get_policy()
    value_function = agent.get_value_function()

    logging.info("Politique obtenue:")
    logging.info(policy)
    logging.info("Fonction de valeur obtenue:")
    logging.info(value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL - Copie\tests\Monty_hall_Lvl_2\policy\value_iteration_policy_MH2.npz')
    logging.info("Politique et fonctions sauvegardées dans 'value_iteration_policy_MH2.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL - Copie\tests\Monty_hall_Lvl_2\policy\value_iteration_policy_MH2.npz')
    loaded_policy = agent.get_policy()
    loaded_value_function = agent.get_value_function()
    logging.info("Politique chargée:")
    logging.info(loaded_policy)
    logging.info("Fonction de valeur chargée:")
    logging.info(loaded_value_function)

    # Démonstration de la politique
    logging.info("\nDémonstration de la politique:")
    num_test_episodes = 1000
    total_wins = 0
    for _ in range(num_test_episodes):
        state = env.reset()
        done = False
        while not done:
            state_tuple = tuple(state) if isinstance(state, (tuple, list)) else (state,)
            if all(dim < policy.shape[i] for i, dim in enumerate(state_tuple)):
                action = policy[state_tuple]
            else:
                logging.warning(f"State {state_tuple} is out of bounds for policy dimensions {policy.shape}, using default action 0.")
                action = 0  # Utiliser une action par défaut (par exemple, 0) si l'état est hors limites
            state, reward, done, _ = env.step(action)
        total_wins += reward
    win_rate = total_wins / num_test_episodes
    logging.info(f"Taux de victoire sur {num_test_episodes} épisodes: {win_rate:.2%}")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    print(f"État initial: {state}")
    done = False
    while not done:
        door, step = state
        state_tuple = tuple(state) if isinstance(state, (tuple, list)) else (state,)
        if all(dim < policy.shape[i] for i, dim in enumerate(state_tuple)):
            recommended_action = policy[state_tuple]
            print(f"Politique recommandée pour l'état {state}: {'Rester' if recommended_action == 0 else 'Changer'}")
        action = int(input(f"Étape {step + 1}/4: Entrez l'action (0 pour rester, 1 pour changer): "))
        state, reward, done, _ = env.step(action)
        print(f"Action: {'Rester' if action == 0 else 'Changer'}")
        print(f"Nouvel état: {state}")
        if done:
            print(f"Résultat: {'Gagné' if reward == 1 else 'Perdu'}")
            print(f"Récompense: {reward}")

if __name__ == "__main__":
    test_value_iteration()
