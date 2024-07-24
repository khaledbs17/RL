import numpy as np
from gym.spaces import Tuple
import time


class Sarsa:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.obs_space_size = env.observation_space.size
        self.act_space_size = len(env.action_space)

        self.q_table = np.zeros((self.obs_space_size, self.act_space_size))
        self.total_reward = 0
        self.duration = 0
        self.policy_changes = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.act_space_size)
        return np.argmax(self.q_table[state])

    def train(self, num_episodes=1000, track_policy_every=100):
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
                td_target = reward + self.gamma * self.q_table[next_state, next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error
                state = next_state
                action = next_action
                episode_reward += reward

            if episode % track_policy_every == 0:
                self.policy_changes.append(self.get_policy().copy())

            total_reward += episode_reward

        self.total_reward = total_reward / num_episodes
        self.duration = time.time() - start_time

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, total_reward=self.total_reward, duration=self.duration,
                 policy_changes=np.array(self.policy_changes))

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
        self.policy_changes = data['policy_changes']
