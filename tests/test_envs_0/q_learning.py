import numpy as np
from tqdm import tqdm
import secret_envs_wrapper
from utils import load_config, calcul_policy, play_a_game_by_Pi, choose_action, update_Q, observe_R_S_prime, \
    save_results_to_pickle
import time

config_file = "D:\projet_DRL - Copie\config.yaml"


def calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env):
    if not env.is_game_over():
        q_s_prime = [Q[s_prime][a_p] for a_p in available_actions_prime]
        best_move = np.max(q_s_prime)  # Directement obtenir la meilleure récompense après le mouvement
        Q[s][a] += alpha * (reward + gamma * best_move - Q[s][a])
    else:
        Q[s][a] += alpha * reward
    return Q[s][a]


def q_learning(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000):
    Q = {}
    total_reward = 0
    start_time = time.time()

    for _ in tqdm(range(nb_iter)):
        env.reset()
        episode_reward = 0

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions()
            Q = update_Q(Q, s, available_actions, env)
            a = choose_action(Q, s, available_actions, epsilon)
            reward, s_prime, available_actions_prime = observe_R_S_prime(env, a)
            Q = update_Q(Q, s_prime, available_actions_prime, env)

            if not env.is_game_over():
                Q[s][a] = calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env)
            else:
                Q[s][a] = Q[s][a] + alpha * reward

            episode_reward += reward

        total_reward += episode_reward

    training_duration = time.time() - start_time
    return Q, total_reward, training_duration

def play_game(game, parameters, results_path, algorithm_name):
    config = None
    if game not in ["SecretEnv0", "SecretEnv1", "SecretEnv2"]:
        config = load_config(config_file, game)

    print(f"Loaded config: {config}")  # Debug print

    alpha = parameters["alpha"]
    epsilon = parameters["epsilon"]
    gamma = parameters["gamma"]
    nb_iter = parameters["nb_iter"]
    n_planning = parameters["n_planning"]

    match game:
        case "SecretEnv0":
            env = secret_envs_wrapper.SecretEnv0()
        case "SecretEnv1":
            env = secret_envs_wrapper.SecretEnv1()
        case "SecretEnv2":
            env = secret_envs_wrapper.SecretEnv2()
        case _:
            print("Game not found")
            return 0

    print(f"Environment created: {env}")
    print(f"Environment type: {type(env)}")
    print(f"num_states method exists: {'num_states' in dir(env)}")
    print(f"Methods available: {dir(env)}")

    Q_optimal, total_reward, training_duration = q_learning(env, alpha, epsilon, gamma, nb_iter)
    Pi = calcul_policy(Q_optimal)
    env.reset()
    save_results_to_pickle(Q_optimal, Pi, results_path, total_reward=total_reward, training_duration=training_duration)
    play_a_game_by_Pi(env, Pi, algorithm_name, game)


if __name__ == '__main__':
    game = "SecretEnv0"
    algorithm_name = "q_learning"
    parameters = {
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 0.999,
        "nb_iter": 10000,
        "n_planning": 10
    }
    results_path = f"D:/projet_DRL - Copie/tests/test2/results/{game}_q_learning.pkl"
    play_game(game, parameters, results_path, algorithm_name)
