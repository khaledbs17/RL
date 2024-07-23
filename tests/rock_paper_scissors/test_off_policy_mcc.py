import sys
import os
import logging

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.rock_paper_scissors import RockPaperScissors
from algorithms.off_policy_mcc import OffPolicyMCC

def test_off_policy_mcc():
    logging.info("Initializing environment...")
    env = RockPaperScissors()
    logging.info("Initializing agent...")
    agent = OffPolicyMCC(env)
    logging.info("Starting training...")
    agent.train(num_episodes=10000)
    
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()
    
    logging.info("Politique obtenue:")
    logging.info(policy)
    logging.info("Fonction de valeur-action obtenue:")
    logging.info(action_value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL\tests\rock_paper_scissors\policy\off_policy_mcc_policy_rps.npz')
    logging.info("Politique et fonctions sauvegardées dans 'off_policy_mcc_policy_rps.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL\tests\rock_paper_scissors\policy\off_policy_mcc_policy_rps.npz')
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
    while steps < max_steps:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        env.render()
        logging.info(f"Action: {action}, État: {state}, Récompense: {reward}")
        steps += 1
        if done:
            break
    if steps >= max_steps:
        logging.info("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    logging.info("\nInteraction manuelle:")
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = int(input("Entrez l'action (0 pour Rock, 1 pour Paper, 2 pour Scissors): "))
        state, reward, done, _ = env.step(action)
        env.render()
        logging.info(f"État: {state}, Récompense: {reward}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_off_policy_mcc()
