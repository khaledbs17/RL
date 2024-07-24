from tqdm import tqdm
import secret_envs_wrapper
from utils import load_config, calcul_policy, play_a_game_by_Pi, choose_action, update_Q, observe_R_S_prime, \
    save_results_to_pickle
import time

config_file = "D:\projet_DRL - Copie\config.yaml"


def sarsa(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 500):
    Q = {}
    total_reward = 0
    start_time = time.time()

    for _ in tqdm(range(nb_iter)):
        env.reset()
        s = env.state_id()
        available_actions = env.available_actions()
        Q = update_Q(Q, s, available_actions, env)
        a = choose_action(Q, s, available_actions, epsilon)
        episode_reward = 0

        while not env.is_game_over():
            reward, s_prime, available_actions_prime = observe_R_S_prime(env, a)
            Q = update_Q(Q, s_prime, available_actions_prime, env)
            a_prime = choose_action(Q, s_prime, available_actions_prime, epsilon)

            if not env.is_game_over():
                q_s_prime = Q[s_prime][a_prime]
                Q[s][a] = Q[s][a] + alpha * (reward + gamma * q_s_prime - Q[s][a])
            else:
                Q[s][a] = Q[s][a] + alpha * reward

            s = s_prime
            a = a_prime
            episode_reward += reward

        total_reward += episode_reward

    training_duration = time.time() - start_time
    return Q, total_reward, training_duration


def play_game(game, parameters, results_path, algorithm_name):
    if "SecretEnv" not in game:
        config = load_config(config_file, game)
    alpha = parameters["alpha"]
    epsilon = parameters["epsilon"]
    gamma = parameters["gamma"]
    nb_iter = parameters["nb_iter"]
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
    Q_optimal, total_reward, training_duration = sarsa(env, alpha, epsilon, gamma, nb_iter)
    Pi = calcul_policy(Q_optimal)
    env.reset()
    save_results_to_pickle(Q_optimal, Pi, results_path, total_reward=total_reward, training_duration=training_duration)
    play_a_game_by_Pi(env, Pi, algorithm_name, game)


if __name__ == '__main__':
    game = "SecretEnv0"
    algorithm_name="sarsa"
    parameters = {"alpha": 0.1,
                  "epsilon": 0.1,
                  "gamma": 0.999,
                  "nb_iter": 10000
                  }
    results_path = f"D:/projet_DRL - Copie/tests/test2/results/{game}_sarsa.pkl"
    play_game(game, parameters, results_path, algorithm_name)
