import sys
import os
import logging
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.rock_paper_scissors import RockPaperScissors
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.off_policy_mcc import OffPolicyMCC
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC
from algorithms.q_learning import QLearning
from algorithms.dyna_q import DynaQ
from algorithms.sarsa import Sarsa
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration

def run_experiment(agent_class, env, num_episodes):
    agent = agent_class(env)
    start_time = time.time()

    if agent_class in [PolicyIteration, ValueIteration]:
        agent.train()
        value_function = agent.get_value_function()
        total_reward = sum(value_function)
    else:
        agent.train(num_episodes=num_episodes)
        total_reward = sum(agent.get_action_value_function().max(axis=1))  # Adjust as needed

    duration = time.time() - start_time
    return total_reward, duration

def main():
    logging.basicConfig(level=logging.INFO)
    env = RockPaperScissors()
    num_episodes = 10000

    algorithms = {
        "MonteCarloES": MonteCarloES,
        "OffPolicyMCC": OffPolicyMCC,
        "OnPolicyFirstVisitMCC": OnPolicyFirstVisitMCC,
        "QLearning": QLearning,
        "DynaQ": DynaQ,
        "Sarsa": Sarsa,
        "PolicyIteration": PolicyIteration,
        "ValueIteration": ValueIteration,
    }

    results = {}

    for name, agent_class in algorithms.items():
        logging.info(f"Running {name}...")
        try:
            total_reward, duration = run_experiment(agent_class, env, num_episodes)
            results[name] = {
                "total_reward": total_reward,
                "duration": duration,
            }
            logging.info(f"{name}: Total Reward = {total_reward}, Duration = {duration}s")
        except Exception as e:
            logging.error(f"Error running {name}: {e}")
            results[name] = {
                "total_reward": None,
                "duration": None,
            }

    np.savez(r"D:\projet_DRL\tests\rock_paper_scissors\policy\algorithm_performance_rps.npz", **results)

if __name__ == "__main__":
    main()
