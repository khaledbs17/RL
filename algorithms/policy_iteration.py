import numpy as np
import logging
import time
from gym.spaces import Tuple

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.obs_space_size = env.observation_space.size
        self.act_space_size = len(env.action_space)
        self.policy = np.zeros(self.obs_space_size, dtype=int)
        self.value_table = np.zeros(self.obs_space_size)
        self.gamma = 0.99
        self.theta = 1e-6
        self.total_reward = 0
        self.duration = 0
        self.policy_changes = []

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.obs_space_size):
                v = self.value_table[state]
                action = self.policy[state]
                self.value_table[state] = sum([prob * (reward + self.gamma * self.value_table[next_state])
                                               for prob, next_state, reward, done in self.env.P[state][action]])
                delta = max(delta, abs(v - self.value_table[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.obs_space_size):
            old_action = self.policy[state]
            action_values = np.zeros(self.act_space_size)
            for action in range(self.act_space_size):
                action_values[action] = sum([prob * (reward + self.gamma * self.value_table[next_state])
                                             for prob, next_state, reward, done in self.env.P[state][action]])
            new_action = np.argmax(action_values)
            self.policy[state] = new_action
            if old_action != new_action:
                policy_stable = False
                self.policy_changes.append(state)
        return policy_stable

    def train(self):
        logging.info("Starting training...")
        iteration = 0
        total_reward = 0
        start_time = time.time()
        while True:
            iteration += 1
            logging.info(f"Policy Iteration {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        self.duration = time.time() - start_time
        # Calculer le total_reward basé sur la policy obtenue
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy[state]
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
        self.total_reward = total_reward

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_table

    def save(self, filepath):
        np.savez(filepath, policy=self.policy, value_function=self.value_table, total_reward=self.total_reward, duration=self.duration, policy_changes=self.policy_changes)

    def load(self, filepath):
        data = np.load(filepath)
        self.policy = data['policy']
        self.value_table = data['value_function']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
        self.policy_changes = data['policy_changes']
