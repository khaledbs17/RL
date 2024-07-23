import numpy as np

class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start[0] * width + start[1]
        self.action_space = np.array([0, 1, 2, 3])  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = np.arange(width * height)
        self.P = self._create_transition_probabilities()

    def _create_transition_probabilities(self):
        P = {}
        for y in range(self.height):
            for x in range(self.width):
                state = y * self.width + x
                P[state] = {a: [] for a in self.action_space}
                if (y, x) in self.obstacles:
                    continue
                for action in self.action_space:
                    next_state, reward, done = self._take_action((y, x), action)
                    prob = 1.0
                    P[state][action].append((prob, next_state, reward, done))
        return P

    def _take_action(self, state, action):
        y, x = state
        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(self.height - 1, y + 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.width - 1, x + 1)

        next_state = y * self.width + x
        if (y, x) in self.obstacles:
            next_state = state[0] * self.width + state[1]

        reward = -0.1  # Une petite pénalité pour chaque mouvement
        if (y, x) == self.goal:
            reward = 1.0
            done = True
        else:
            done = False

        return next_state, reward, done

    def reset(self):
        self.state = self.start[0] * self.width + self.start[1]
        return self.state

    def step(self, action):
        next_state, reward, done = self._take_action(divmod(self.state, self.width), action)
        self.state = next_state
        return self.state, reward, done, {}

    def render(self):
        grid = [['-' for _ in range(self.width)] for _ in range(self.height)]
        for y, x in self.obstacles:
            grid[y][x] = 'X'
        sy, sx = self.start
        gy, gx = self.goal
        grid[sy][sx] = 'A'
        grid[gy][gx] = 'G'
        state_y, state_x = divmod(self.state, self.width)
        grid[state_y][state_x] = 'O'
        for row in grid:
            print(' '.join(row))
        print()  # Pour ajouter une ligne vide entre les rendus
