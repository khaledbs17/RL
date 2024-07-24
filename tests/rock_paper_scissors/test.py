import sys
import os
import logging
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.rock_paper_scissors import TwoRoundRockPaperScissors
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.off_policy_mcc import OffPolicyMCC
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC
from algorithms.q_learning import QLearning
from algorithms.dyna_q import DynaQ
from algorithms.sarsa import Sarsa
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration

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
    logging.basicConfig(level=logging.INFO)
    env = TwoRoundRockPaperScissors()
    num_episodes = 1000
    results = {}

    policies = {
        'MonteCarloES': r'./tests/rock_paper_scissors/best_policy/best_monte_carlo_es_policy_rps.npz',
        'OffPolicyMCC': r'./tests/rock_paper_scissors/best_policy/best_off_policy_mcc_policy_rps.npz',
        'OnPolicyFirstVisitMCC': r'./tests/rock_paper_scissors/best_policy/best_on_policy_first_visit_mcc_policy_rps.npz',
        'QLearning': r'./tests/rock_paper_scissors/best_policy/best_q_learning_policy_rps.npz',
        'DynaQ': r'./tests/rock_paper_scissors/best_policy/best_dyna_q_policy_rps.npz',
        'Sarsa': r'./tests/rock_paper_scissors/best_policy/best_sarsa_policy_rps.npz',
        'PolicyIteration': r'./tests/rock_paper_scissors/best_policy/best_policy_iteration_policy_rps.npz',
        'ValueIteration': r'./tests/rock_paper_scissors/best_policy/best_value_iteration_policy_rps.npz',
    }

    algorithms = {
        'MonteCarloES': MonteCarloES,
        'OffPolicyMCC': OffPolicyMCC,
        'OnPolicyFirstVisitMCC': OnPolicyFirstVisitMCC,
        'QLearning': QLearning,
        'DynaQ': DynaQ,
        'Sarsa': Sarsa,
        'PolicyIteration': PolicyIteration,
        'ValueIteration': ValueIteration,
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
    np.savez(r'./tests/rock_paper_scissors/best_policy/algorithm_performance_rps.npz', **results)

if __name__ == "__main__":
    main()
