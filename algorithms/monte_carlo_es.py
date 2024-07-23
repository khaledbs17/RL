import numpy as np
import random
from gym.spaces import Tuple
import time

class MonteCarloES:
    def __init__(self, env):
        self.env = env
        if isinstance(env.observation_space, Tuple):
            obs_space_size = tuple(space.n for space in env.observation_space.spaces)
        else:
            obs_space_size = (env.observation_space.size,)

        act_space_size = env.action_space.size
        self.q_table = np.zeros(obs_space_size + (act_space_size,))
        self.returns = {}
        self.policy = np.zeros(obs_space_size, dtype=int)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.total_reward = 0
        self.duration = 0

    def state_to_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(map(int, state))
        else:
            return (int(state),)

    # Création d'un épisode basé sur la politique actuelle
    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        step_counter = 0  # Compteur pour limiter le nombre d'étapes
        max_steps = 100  # Limite maximale des étapes pour éviter les boucles infinies

        while not done and step_counter < max_steps:
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(self.env.action_space)  # Exploration aléatoire
            else:
                action = self.policy[self.state_to_tuple(state)]  # Exploitation de la politique actuelle

            next_state, reward, done, _ = self.env.step(action)
            episode.append((self.state_to_tuple(state), action, reward))
            state = next_state
            step_counter += 1
            print(f"Step {step_counter}: state={state}, action={action}, reward={reward}, done={done}")

        if step_counter >= max_steps:
            print("Maximum steps reached in generate_episode")

        return episode

    # Mise à jour de la politique et de la fonction de valeur
    def train(self, num_episodes):
        total_reward = 0
        start_time = time.time()
        for episode_num in range(num_episodes):
            print(f"Generating episode {episode_num + 1}/{num_episodes}...")
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any((s == state and a == action) for (s, a, r) in episode[:t]):
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    self.returns[(state, action)].append(G)
                    self.q_table[state][action] = np.mean(self.returns[(state, action)])
                    self.policy[state] = np.argmax(self.q_table[state])
            print(f"Episode {episode_num + 1}/{num_episodes} completed.")
        self.total_reward = total_reward / num_episodes
        self.duration = time.time() - start_time

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, policy=self.policy, total_reward=self.total_reward, duration=self.duration)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.total_reward = data.get('total_reward', 0)
        self.duration = data.get('duration', 0)
