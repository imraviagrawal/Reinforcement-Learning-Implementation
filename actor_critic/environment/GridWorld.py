# Imports
import numpy as np


# class to create grid world and perform actions
class gridWorld():
    def __init__(self):
        self.name = "grid"
        self.i = 0 # initial position
        self.j = 0 # initial position
        self.action = [0, 1, 2, 3]
        self.world =  [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -10, 0, 10]] # inializing the world and values as a rewards
        self.actions = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # [right, down, left, up]
        # example for right action, We can have take (right, down, up or stay in the same location) This dictionary map the action and possible
        # env dynamics we may have to face

        self.EnvDynamicsActionMapping = {(0, 1): [(0, 1), (1, 0), (-1, 0), (0, 0)],
                                    (1, 0): [[1, 0], [0, 1], [0, -1], [0, 0]],
                                    (0, -1): [[0, -1], [1, 0], [-1, 0], [0, 0]],
                                    (-1, 0): [[-1, 0], [1, 0], [-1, 0], [0, 0]]}

        self.envDynamics = [0.8, 0.05, 0.05, 0.1]  # Environment dynamics
        self.status = False

    def performAction(self, action):
        # choicing action
        action = self.actions[action]

        # Applying environment dynamics
        action_index = np.random.choice([0, 1, 2, 3], p=self.envDynamics)  # choosing the action by env dynamics
        i_, j_ = self.EnvDynamicsActionMapping[action][action_index]  # getting the action index after the env dynamics

        # Checking if the state are within bound and not hitting the wall
        if self.i + i_ >= 0 and self.i + i_ < len(self.world) and self.j + j_ >= 0 and self.j + j_ < len(self.world[0]):
            temp_i, temp_j = self.i + i_, self.j + j_
            if ((temp_i, temp_j) == (2, 2)) or ((temp_i, temp_j) == (3, 2)):
                self.i = self.i
                self.j = self.j
            else:
                self.i = temp_i
                self.j = temp_j

        # reached the terminal state
        if self.i == 4 and self.j == 4:
            self.status = True

        reward = self.world[self.i][self.j]
        return (self.i, self.j), reward, self.status

    def reset(self):
        self.i = 0
        self.j = 0
        self.status = False
        return (self.i, self.j)


