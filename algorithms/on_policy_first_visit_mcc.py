import numpy as np
import time
from gym.spaces import Tuple


class OnPolicyFirstVisitMCC:
    def __init__(self, env):
        self.env = env
        if isinstance(env.observation_space, Tuple):
            obs_space_size = tuple(space.n for space in env.observation_space.spaces)
        else:
            obs_space_size = (env.observation_space.size,)

        act_space_size = len(env.action_space)
        self.q_table = np.zeros(obs_space_size + (act_space_size,))
        self.returns = {}
        self.policy = np.zeros(obs_space_size, dtype=int)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.total_reward = 0
        self.duration = 0
        self.policy_changes = []

    def state_to_tuple(self, state):
        if isinstance(state, (tuple, list)):
            return tuple(map(int, state))
        else:
            return (int(state),)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # Exploration
            return int(np.random.choice(self.env.action_space))
        else:  # Exploitation
            return int(self.policy[self.state_to_tuple(state)])

    def generate_episode(self, max_steps):
        episode = []
        state = self.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((self.state_to_tuple(state), action, reward))
            state = next_state
            steps += 1
        return episode

    def train(self, num_episodes, max_steps_per_episode=1000):
        total_reward = 0
        start_time = time.time()
        for episode_num in range(num_episodes):
            episode = self.generate_episode(max_steps_per_episode)
            G = 0
            episode_reward = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any((s == state and a == action) for (s, a, r) in episode[:t]):
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    self.returns[(state, action)].append(G)
                    old_action = self.policy[state]
                    self.q_table[state][action] = np.mean(self.returns[(state, action)])
                    self.policy[state] = int(np.argmax(self.q_table[state]))
                    if old_action != self.policy[state]:
                        self.policy_changes.append(episode_num)
                episode_reward += reward
            total_reward += episode_reward
            print(f"Episode {episode_num + 1}/{num_episodes} completed.")
        self.total_reward = total_reward / num_episodes
        self.duration = time.time() - start_time

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, policy=self.policy, total_reward=self.total_reward, duration=self.duration, policy_changes=self.policy_changes)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
        self.policy_changes = data['policy_changes']
