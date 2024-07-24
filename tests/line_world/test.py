import sys
import os
import time
import numpy as np
import logging

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.line_world import LineWorld
from algorithms.dyna_q import DynaQ
from algorithms.q_learning import QLearning
from algorithms.sarsa import Sarsa
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.off_policy_mcc import OffPolicyMCC
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration

logging.basicConfig(level=logging.INFO)

def load_best_policy(agent_class, env, filepath):
    agent = agent_class(env)
    agent.load(filepath)
    return agent.get_policy()

def run_experiment(policy, env, num_episodes=1000):
    total_reward = 0
    state = env.reset()
    steps = 0
    while steps < num_episodes:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        steps += 1
    return total_reward

def main():
    env = LineWorld(length=10, start=0, goal=9)
    num_episodes = 1000
    results = {}

    policies = {
        'Dyna-Q': r'tests/line_world/best_policy/best_dyna_q_policy_line_world.npz',
        'Q-Learning': r'tests/line_world/best_policy/best_q_learning_policy_line_world.npz',
        'SARSA': r'tests/line_world/best_policy/best_sarsa_policy.npz',
        'Monte Carlo ES': r'tests/line_world/best_policy/new_parameters_monte_carlo_es_policy.npz',
        'Off-Policy MCC': r'tests/line_world/best_policy/best_off_policy_mcc_policy.npz',
        'On-Policy First Visit MCC': r'tests/line_world/best_policy/best_on_policy_first_visit_mcc_policy.npz',
        'Policy Iteration': r'tests/line_world/best_policy/best_policy_iteration_policy.npz',
        'Value Iteration': r'tests/line_world/best_policy/best_value_iteration_policy.npz'
    }

    algorithms = {
        'Dyna-Q': DynaQ,
        'Q-Learning': QLearning,
        'SARSA': Sarsa,
        'Monte Carlo ES': MonteCarloES,
        'Off-Policy MCC': OffPolicyMCC,
        'On-Policy First Visit MCC': OnPolicyFirstVisitMCC,
        'Policy Iteration': PolicyIteration,
        'Value Iteration': ValueIteration
    }

    for name, filepath in policies.items():
        logging.info(f"Loading policy for {name}...")
        policy = load_best_policy(algorithms[name], env, filepath)
        logging.info(f"Running experiment for {name}...")
        start_time = time.time()
        total_reward = run_experiment(policy, env, num_episodes)
        end_time = time.time()
        duration = end_time - start_time
        results[name] = {
            'total_reward': total_reward,
            'duration': duration
        }
        logging.info(f"{name} - Total Reward: {total_reward}, Duration: {duration} seconds")

    # Sauvegarde des résultats
    np.savez(r'tests/line_world/best_policy/experiment_results.npz', **results)

if __name__ == "__main__":
    main()
