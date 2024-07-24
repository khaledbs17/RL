import sys
import os
import logging

# Assurez-vous que le chemin vers vos modules personnalisés est correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.secret_envs_wrapper import SecretEnv0
from algorithms.policy_iteration import PolicyIteration

logging.basicConfig(level=logging.INFO)

def test_policy_iteration():
    logging.info("Initializing environment...")
    env = SecretEnv0()

    logging.info("Initializing agent...")
    agent = PolicyIteration(env)

    logging.info("Starting training...")
    agent.train()

    policy = agent.get_policy()
    value_function = agent.get_value_function()

    logging.info("Policy obtained:")
    logging.info(policy)

    logging.info("Value function obtained:")
    logging.info(value_function)

    # Sauvegarde de la politique et de la fonction de valeur
    agent.save('policy_iteration_secret_env_0.npz')
    logging.info("Policy and value function saved in 'policy_iteration_secret_env_0.npz'.")

    # Chargement et affichage de la politique et de la fonction de valeur
    agent.load('policy_iteration_secret_env_0.npz')
    loaded_policy = agent.get_policy()
    loaded_value_function = agent.get_value_function()

    logging.info("Loaded policy:")
    logging.info(loaded_policy)
    logging.info("Loaded value function:")
    logging.info(loaded_value_function)

    # Démonstration de la politique
    state = env.reset()
    env.display()
    while not env.is_game_over():
        action = loaded_policy[state]
        env.step(action)
        env.display()
        state = env.state_id()
        logging.info(f"Action: {action}, State: {state}, Score: {env.score()}")

if __name__ == "__main__":
    test_policy_iteration()
