import numpy as np
from gym.spaces import Discrete


class MontyHallLevel1:
    def __init__(self):
        self.observation_space = Discrete(3)  # 3 doors
        self.action_space = Discrete(2)  # Stick or Switch
        self.state = self.reset()
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self):
        P = {}
        for state in range(self.observation_space.n):
            P[state] = {a: [] for a in range(self.action_space.n)}
            for action in range(self.action_space.n):
                for winning_door in range(self.observation_space.n):
                    next_state = 0 if action == 0 else (winning_door if winning_door != state else (state + 1) % 3)
                    reward = 1.0 if next_state == winning_door else 0.0
                    done = True
                    prob = 1 / self.observation_space.n
                    P[state][action].append((prob, next_state, reward, done))
        return P

    def reset(self):
        self.state = np.random.choice([0, 1, 2])
        return self.state

    def step(self, action):
        winning_door = np.random.choice([0, 1, 2])
        next_state = 0 if action == 0 else (winning_door if winning_door != self.state else (self.state + 1) % 3)
        reward = 1.0 if next_state == winning_door else 0.0
        self.state = next_state
        done = True
        return self.state, reward, done, {}

    def render(self):
        print(f"State: {self.state}")
