import numpy as np
from gym.spaces import Discrete

class RockPaperScissors:
    def __init__(self):
        self.observation_space = Discrete(3)  # 3 states corresponding to Rock, Paper, Scissors
        self.action_space = Discrete(3)  # Rock, Paper, Scissors
        self.state = 0
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self):
        P = {}
        for state in range(self.observation_space.n):
            P[state] = {a: [] for a in range(self.action_space.n)}
            for action in range(self.action_space.n):
                for opponent_action in range(self.action_space.n):
                    next_state = opponent_action
                    reward = self._get_reward(action, opponent_action)
                    done = True
                    prob = 1 / self.action_space.n  # opponent plays randomly
                    P[state][action].append((prob, next_state, reward, done))
        return P

    def _get_reward(self, action, opponent_action):
        if action == opponent_action:
            return 0
        elif (action == 0 and opponent_action == 2) or \
             (action == 1 and opponent_action == 0) or \
             (action == 2 and opponent_action == 1):
            return 1
        else:
            return -1

    def reset(self):
        self.state = np.random.choice([0, 1, 2])
        return self.state

    def step(self, action):
        opponent_action = np.random.choice([0, 1, 2])
        reward = self._get_reward(action, opponent_action)
        self.state = opponent_action
        done = True  # Each game consists of a single round
        return self.state, reward, done, {}

    def render(self):
        print(f"State: {self.state}")
