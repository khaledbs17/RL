import numpy as np
import random
import time
from gym.spaces import Tuple


class MonteCarloES:
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        obs_space_size = (env.width * env.height,)
        act_space_size = env.action_space.size

        self.q_table = np.zeros(obs_space_size + (act_space_size,))
        self.returns = {}
        self.policy = np.zeros(obs_space_size, dtype=int)
        self.total_reward = 0
        self.duration = 0
        self.policy_changes = []

    def state_to_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(map(int, state))
        else:
            return (int(state),)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        step_counter = 0
        max_steps = 100

        while not done and step_counter < max_steps:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.sample()  # Exploration aléatoire
            else:
                action = self.policy[self.state_to_tuple(state)]  # Exploitation de la politique actuelle

            next_state, reward, done, _ = self.env.step(action)
            episode.append((self.state_to_tuple(state), action, reward))
            state = next_state
            step_counter += 1

        return episode

    def train(self, num_episodes):
        start_time = time.time()
        previous_policy = np.copy(self.policy)
        for episode_num in range(num_episodes):
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

            if episode_num % 10 == 0:
                policy_change = np.sum(previous_policy != self.policy)
                self.policy_changes.append(policy_change)
                previous_policy = np.copy(self.policy)
                print(f"Episode {episode_num + 1}/{num_episodes}, Policy changes: {policy_change}")

        self.duration = time.time() - start_time

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, policy=self.policy, total_reward=self.total_reward,
                 duration=self.duration, policy_changes=self.policy_changes)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
        self.policy_changes = data['policy_changes']
