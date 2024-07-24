import numpy as np
from gym.spaces import Discrete, Tuple


class MontyHallLevel2:
    def __init__(self):
        self.observation_space = Tuple((Discrete(5), Discrete(4)))  # (porte choisie, étape)
        self.action_space = Discrete(2)  # 0: Stick, 1: Switch
        self.state = None
        self.winning_door = None
        self.step_count = 0
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self):
        P = {}
        for door in range(5):
            for step in range(4):
                state = (door, step)
                P[state] = {a: [] for a in range(self.action_space.n)}

                if step < 3:
                    # Les 3 premières actions n'ont pas d'effet réel
                    for action in range(self.action_space.n):
                        P[state][action].append((1.0, (door, step + 1), 0.0, False))
                else:
                    # Dernière action (4ème étape)
                    for action in range(self.action_space.n):
                        for winning_door in range(5):
                            if action == 0:  # Stick
                                next_door = door
                            else:  # Switch
                                available_doors = [i for i in range(5) if i != door]
                                next_door = np.random.choice(available_doors)

                            reward = 1.0 if next_door == winning_door else 0.0
                            prob = 1 / 5  # Probabilité égale pour chaque porte d'être gagnante
                            P[state][action].append((prob, (next_door, 3), reward, True))

        return P

    def reset(self):
        self.winning_door = np.random.randint(5)
        self.step_count = 0
        initial_door = np.random.randint(5)
        self.state = (initial_door, self.step_count)
        return self.state

    def step(self, action):
        current_door, current_step = self.state

        if current_step < 3:
            # Les 3 premières actions n'ont pas d'effet réel
            self.step_count += 1
            self.state = (current_door, self.step_count)
            return self.state, 0.0, False, {}
        else:
            # Dernière action (4ème)
            if action == 0:  # Stick
                final_door = current_door
            else:  # Switch
                available_doors = [i for i in range(5) if i != current_door]
                final_door = np.random.choice(available_doors)

            reward = 1.0 if final_door == self.winning_door else 0.0
            self.state = (final_door, 3)
            return self.state, reward, True, {}

    def render(self):
        door, step = self.state
        print(f"Étape: {step + 1}/4, Porte actuellement choisie: {door}")
        if step == 3:
            print(f"Porte gagnante: {self.winning_door}")
            print(f"Résultat: {'Gagné' if door == self.winning_door else 'Perdu'}")