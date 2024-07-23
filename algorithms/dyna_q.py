import numpy as np
import random
import time
from gym.spaces import Tuple

class DynaQ:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        if isinstance(env.observation_space, Tuple):
            obs_space_size = tuple(space.n for space in env.observation_space.spaces)
        else:
            obs_space_size = (env.observation_space.size,)  # Utiliser .size pour obtenir la taille de l'espace d'observation

        act_space_size = env.action_space.size  # Utiliser .size pour obtenir la taille de l'espace d'action
        self.q_table = np.zeros(obs_space_size + (act_space_size,))
        self.model = {}
        self.total_reward = 0
        self.duration = 0

    def state_to_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(map(int, state))
        else:
            return (int(state),)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.env.action_space))  # Choisir une action aléatoire
        else:
            return int(np.argmax(self.q_table[self.state_to_tuple(state)]))  # Assurer que l'action est un entier

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[self.state_to_tuple(next_state)])
        td_target = reward + self.gamma * self.q_table[self.state_to_tuple(next_state)][best_next_action]
        self.q_table[self.state_to_tuple(state)][action] += self.alpha * (td_target - self.q_table[self.state_to_tuple(state)][action])
        self.model[(self.state_to_tuple(state), action)] = (self.state_to_tuple(next_state), reward)

    def planning(self):
        for _ in range(self.planning_steps):
            state, action = random.choice(list(self.model.keys()))
            next_state, reward = self.model[(state, action)]
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
            self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def train(self, num_episodes):
        total_reward = 0
        start_time = time.time()
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state)
                print(f"Chosen action: {action}, Type: {type(action)}")  # Debugging print
                action = int(action)  # Convertir explicitement l'action en int
                if not isinstance(action, int):
                    raise ValueError("Action must be an integer")
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                self.planning()
                state = next_state
                episode_reward += reward
            total_reward += episode_reward
        self.total_reward = total_reward / num_episodes
        self.duration = time.time() - start_time

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, total_reward=self.total_reward, duration=self.duration)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
