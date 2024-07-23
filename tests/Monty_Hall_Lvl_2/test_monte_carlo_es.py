import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.monty_hall_level_2 import MontyHallLevel2
from algorithms.monte_carlo_es import MonteCarloES


def test_monte_carlo_es():
    print("Initializing environment...")
    env = MontyHallLevel2()
    print("Initializing agent...")
    agent = MonteCarloES(env)
    print("Starting training...")
    agent.train(num_episodes=100000)  # Augmenté à 100000 épisodes
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL - Copie\tests\Monty_hall_Lvl_2\policy\monte_carlo_es_policy_MH2.npz')
    print("Politique et fonctions sauvegardées dans 'monte_carlo_es_policy_MH2.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL - Copie\tests\Monty_hall_Lvl_2\policy\monte_carlo_es_policy_MH2.npz')
    loaded_policy = agent.get_policy()
    loaded_action_value_function = agent.get_action_value_function()
    print("Politique chargée:")
    print(loaded_policy)
    print("Fonction de valeur-action chargée:")
    print(loaded_action_value_function)

    # Démonstration de la politique
    print("\nDémonstration de la politique:")
    num_episodes = 1000
    total_wins = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            door, step = state
            # Vérifiez que door et step sont dans les limites de la politique
            if door < policy.shape[0] and step < policy.shape[1]:
                action = policy[door, step]
            else:
                print(f"State {state} is out of bounds for policy dimensions {policy.shape}, using default action 0.")
                action = 0  # Utiliser une action par défaut (par exemple, 0) si l'état est hors limites
            state, reward, done, _ = env.step(action)
        total_wins += reward
    win_rate = total_wins / num_episodes
    print(f"Taux de victoire sur {num_episodes} épisodes: {win_rate:.2%}")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    print(f"État initial: {state}")
    done = False
    while not done:
        door, step = state
        # Afficher l'action recommandée par la politique
        if door < policy.shape[0] and step < policy.shape[1]:
            recommended_action = policy[door, step]
            print(f"Politique recommandée pour l'état {state}: {'Rester' if recommended_action == 0 else 'Changer'}")
        else:
            print(f"State {state} is out of bounds for policy dimensions {policy.shape}, using default action 0.")
            recommended_action = 0

        action = int(input(f"Étape {step + 1}/4: Entrez l'action (0 pour rester, 1 pour changer): "))
        state, reward, done, _ = env.step(action)
        print(f"Action: {'Rester' if action == 0 else 'Changer'}")
        print(f"Nouvel état: {state}")
        if done:
            print(f"Résultat: {'Gagné' if reward == 1 else 'Perdu'}")
            print(f"Récompense: {reward}")
    print("Fin de l'interaction manuelle.")


if __name__ == "__main__":
    test_monte_carlo_es()
