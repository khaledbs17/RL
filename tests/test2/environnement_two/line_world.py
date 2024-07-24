from gym.spaces import Discrete
import numpy as np

class LineWorld:
    def __init__(self, length, start, goal):
        print(f"Initializing LineWorld with length={length}, start={start}, goal={goal}")
        self.length = length
        self.start = start
        self.goal = goal
        self.state = start
        self.observation_space = Discrete(length)
        self.action_space = Discrete(2)
        self.P = self._build_transition_probabilities()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.item()
            else:
                raise ValueError("Action array must have size 1")
        action = int(action)
        if action == 0:  # gauche
            self.state = max(0, self.state - 1)
        elif action == 1:  # droite
            self.state = min(self.length - 1, self.state + 1)

        reward = -0.01
        done = False
        if self.state == self.goal:
            reward = 1.0
            done = True

        return self.state, reward, done, {}

    def render(self):
        world = ['-'] * self.length
        world[self.goal] = 'G'
        world[self.state] = 'O'
        print(''.join(world))

    def _build_transition_probabilities(self):
        P = {state: {action: [] for action in range(self.action_space.n)} for state in range(self.observation_space.n)}
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                if action == 0:  # gauche
                    next_state = max(0, state - 1)
                else:  # droite
                    next_state = min(self.length - 1, state + 1)
                reward = 1.0 if next_state == self.goal else -0.01
                done = next_state == self.goal
                P[state][action].append((1.0, next_state, reward, done))
        return P

    def num_states(self):
        return self.length

    def num_actions(self):
        return self.action_space.n

    def state_id(self):
        return self.state

    def available_actions(self):
        return list(range(self.action_space.n))

    def is_game_over(self):
        return self.state == self.goal

    def score(self):
        return 1.0 if self.state == self.goal else 0.0