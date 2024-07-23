import sys
import os
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.line_world import LineWorld
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC


def test_on_policy_first_visit_mcc():
    logging.info("Initializing environment...")
    env = LineWorld(length=10, start=0, goal=9)
    logging.info("Initializing agent...")
    agent = OnPolicyFirstVisitMCC(env)
    logging.info("Starting training...")
    agent.train(num_episodes=10000)  # Réduire le nombre d'épisodes pour les tests
    logging.info("Training completed.")
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    logging.info("Politique obtenue:")
    logging.info(policy)
    logging.info("Fonction de valeur-action obtenue:")
    logging.info(action_value_function)

    # Sauvegarde des résultats
    agent.save(r'D:\projet_DRL - Copie\tests\line_world\policy\on_policy_first_visit_mcc_policy.npz')
    logging.info("Politique et fonctions sauvegardées dans 'on_policy_first_visit_mcc_policy.npz'.")

    # Chargement des résultats
    agent.load(r'D:\projet_DRL - Copie\tests\line_world\policy\on_policy_first_visit_mcc_policy.npz')
    logging.info("Politique chargée:")
    logging.info(agent.get_policy())
    logging.info("Fonction de valeur-action chargée:")
    logging.info(agent.get_action_value_function())

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
        print(f"Action: {action}, État: {state}, Récompense: {reward}")
        steps += 1
    if steps >= max_steps:
        logging.info("Limite de pas atteinte, la politique peut ne pas être optimale.")

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
    test_on_policy_first_visit_mcc()
