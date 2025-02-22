import yaml
import random
import pickle
import numpy as np
import os


def load_config(config_file, env_name):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config[env_name]


def calcul_policy(Q):
    Pi = {}
    if type(Q) == dict:
        for s in Q.keys():
            best_a = None
            best_a_score = 0.0

            for a, a_score in Q[s].items():
                if best_a is None or best_a_score <= a_score:
                    best_a = a
                    best_a_score = a_score

            Pi[s] = best_a
        return Pi
    else:
        for s in range(Q.shape[0]):
            best_a = None
            best_a_score = 0.0

            for a in range(Q.shape[1]):
                a_score = Q[s, a]
                if best_a is None or best_a_score <= a_score:
                    best_a = a
                    best_a_score = a_score

            Pi[s] = best_a
        return Pi


def play_a_game_by_Pi(env, Pi, algorithm_name, game_name):
    random_move = 0
    move = 0
    while not env.is_game_over():
        move += 1
        if hasattr(env, 'render'):
            env.render()
        elif hasattr(env, 'display'):
            env.display()
        print(f"Score: {env.score()}")
        if env.state_id() in Pi:
            a = Pi[env.state_id()]
            if hasattr(env, 'is_forbidden') and env.is_forbidden(a):
                a = random.choice(env.available_actions())
                env.step(a)
                random_move += 1
            else:
                env.step(a)
        else:
            a = random.choice(env.available_actions())
            env.step(a)
            random_move += 1
    if hasattr(env, 'render'):
        env.render()
    elif hasattr(env, 'display'):
        env.display()
    print(f"Final Score: {env.score()}")
    print(f"Random moves: {random_move} out of {move} total moves")

    # Create a directory for storing policies if it doesn't exist
    policy_dir = 'policies'
    os.makedirs(policy_dir, exist_ok=True)

    # Create the filename with algorithm name and game name
    filename = f'{policy_dir}/{algorithm_name}_{game_name}_policy.pkl'

    with open(filename, 'wb') as fichier:
        pickle.dump(Pi, fichier)
    print(f"Policy saved as {filename}")


def choose_action(Q, s, available_actions, epsilon):
    if random.uniform(0, 1) < epsilon:
        a = random.choice(available_actions)
    else:
        q_s = [Q[s][a] for a in available_actions]
        best_a_index = np.argmax(q_s)
        a = available_actions[best_a_index]
    return a


def update_Q(Q, s, available_actions, env):
    if hasattr(env, 'terminals'):
        if s not in Q:
            Q[s] = {}
            if s in env.terminals:
                for a in available_actions:
                    Q[s][a] = 0
            else:
                for a in available_actions:
                    Q[s][a] = np.random.random()
    else:
        if s not in Q:
            Q[s] = {}
            for a in available_actions:
                Q[s][a] = np.random.random()
    return Q


def save_results_to_pickle(Q, Pi, file_path, total_reward=0, training_duration=0):
    # Crée le dossier si nécessaire
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        'Q': Q,
        'Pi': Pi,
        'total_reward': total_reward,
        'training_duration': training_duration
    }

    # Convertir le tableau numpy en liste pour la sérialisation, si nécessaire
    if isinstance(Q, np.ndarray):
        new_data['Q'] = Q.tolist()

    # Vérifie si le fichier existe déjà
    if os.path.exists(file_path):
        # Charge les données existantes
        with open(file_path, 'rb') as file:
            try:
                existing_data = pickle.load(file)
            except EOFError:  # Gère le cas où le fichier est vide
                existing_data = []
        # Ajoute les nouvelles données
        existing_data.append(new_data)
        data_to_save = existing_data
    else:
        # Si le fichier n'existe pas, prépare les nouvelles données pour la sauvegarde
        data_to_save = [new_data]

    # Sauvegarde les données dans le fichier
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)



def observe_R_S_prime(env, a):
    prev_score = env.score()
    env.step(a)
    new_score = env.score()
    reward = new_score - prev_score
    s_prime = env.state_id()
    available_actions_prime = env.available_actions()
    return reward, s_prime, available_actions_prime
