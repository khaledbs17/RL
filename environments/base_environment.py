class BaseEnvironment:
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.state = None
        self.P = None  # Transition probabilities

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def _build_transition_matrix(self):
        raise NotImplementedError

    def _next_state_reward_done(self, state, action):
        raise NotImplementedError
