import numpy as np
import logging
import time


class ValueIteration:
    def __init__(self, env):
        self.env = env
        obs_space_size = (self.env.width * self.env.height,)

        self.V = np.zeros(obs_space_size)
        self.policy = np.zeros(obs_space_size, dtype=int)
        self.gamma = 0.99
        self.theta = 1e-6
        self.total_reward = 0
        self.duration = 0

    def calculate_value(self, state, action):
        """Calcule la valeur attendue pour un état et une action donnés."""
        total = 0
        for prob, next_state, reward, done in self.env.P[state][action]:
            total += prob * (reward + self.gamma * self.V[next_state])
        return total

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(len(self.V)):
                v = self.V[state]
                self.V[state] = max(
                    [self.calculate_value(state, action) for action in range((self.env.action_space.size))])
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(len(self.V)):
            old_action = self.policy[state]
            action_values = np.zeros((self.env.action_space.size))
            for action in range((self.env.action_space.size)):
                action_values[action] = self.calculate_value(state, action)
            new_action = np.argmax(action_values)
            self.policy[state] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable

    def train(self):
        logging.info("Starting training...")
        iteration = 0
        start_time = time.time()
        while True:
            iteration += 1
            logging.info(f"Iteration: {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        self.duration = time.time() - start_time
        # Calculer le total_reward basé sur la policy obtenue
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.policy[state]
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
        self.total_reward = total_reward

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.V

    def save(self, filepath):
        np.savez(filepath, policy=self.policy, value_function=self.V, total_reward=self.total_reward,
                 duration=self.duration)

    def load(self, filepath):
        data = np.load(filepath)
        self.policy = data['policy']
        self.V = data['value_function']
        self.total_reward = data['total_reward']
        self.duration = data['duration']
