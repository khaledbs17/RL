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


def run_experiment(agent_class, env, num_episodes=10000):
    agent = agent_class(env)
    start_time = time.time()

    if agent_class in [DynaQ, QLearning, Sarsa, MonteCarloES, OffPolicyMCC, OnPolicyFirstVisitMCC]:
        agent.train(num_episodes=num_episodes)
    else:
        agent.train()

    end_time = time.time()

    policy = agent.get_policy()
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
    return total_reward, end_time - start_time


def main():
    env = LineWorld(length=10, start=0, goal=9)
    num_episodes = 1000
    results = {}

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

    for name, agent_class in algorithms.items():
        logging.info(f"Running {name}...")
        total_reward, duration = run_experiment(agent_class, env, num_episodes)
        results[name] = {
            'total_reward': total_reward,
            'duration': duration
        }
        logging.info(f"{name} - Total Reward: {total_reward}, Duration: {duration} seconds")

    # Sauvegarde des résultats
    np.savez(r'D:\projet_DRL - Copie\tests\Monty_Hall_Lvl_1\policy\experiment_results.npz', **results)


if __name__ == "__main__":
    main()
