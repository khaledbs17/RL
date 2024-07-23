import numpy as np
from gym.spaces import Tuple
import time
class Sarsa:
    def __init__(self, env):
        self.env = env
        if isinstance(env.observation_space, Tuple):
            obs_space_size = tuple(space.n for space in env.observation_space.spaces)
        else:
            obs_space_size = (env.observation_space.size,)

        act_space_size = env.action_space.size
        self.q_table = np.zeros(obs_space_size + (act_space_size,))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.total_reward = 0
        self.duration = 0

    def state_to_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(map(int, state))
        else:
            return (int(state),)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.q_table[self.state_to_tuple(state)])

    def train(self, num_episodes):
        total_reward = 0
        start_time = time.time()
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            episode_reward = 0
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.q_table[self.state_to_tuple(state), action] += self.alpha * (
                    reward + self.gamma * self.q_table[self.state_to_tuple(next_state), next_action] - self.q_table[self.state_to_tuple(state), action]
                )
                state, action = next_state, next_action
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
