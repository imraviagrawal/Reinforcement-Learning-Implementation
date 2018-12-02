# Author: Ravi Agrawal
# Import

import numpy as np
import math

class mountaincar():
    def __init__(self, ):
        self.name = "mountain"
        self.maxSpeed = 0.07
        self.minSpeed = -0.07
        self.maxPosition = 0.5
        self.minPosition = -1.2
        self.action = np.array([-1, 0, 1])
        self.probs=[0.33333333, 0.33333333, 0.33333333]
        self.lambda_ = 1.0
        self.state = (-0.5, 0)
        self.status = False

    def performAction(self, action):
        position, velocity = self.state
        velocity += 0.001*action - 0.0025*math.cos(3*position)
        velocity = np.clip(velocity, self.minSpeed, self.maxSpeed)
        position += velocity
        position = np.clip(position, self.minPosition, self.maxPosition)

        if (position == self.minPosition and velocity < 0):
            velocity = 0

        self.status = position >= self.maxPosition

        rewards = -1
        if self.status:
            rewards = 0

        self.state = (position, velocity)
        return self.state, rewards, self.status


    def reset(self):
        self.state = (-0.5, 0)
        self.status = False
        return self.state