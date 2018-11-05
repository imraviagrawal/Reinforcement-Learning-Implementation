"""File to run the gridWorld algorithm
   run instructions needs to be updated
"""

# imports
from TD_algorithm import TD
import numpy as np
from GridWorld import gridWorld
import matplotlib.pyplot as plt
from utils import softmax

# Initializing the gridworld
env = gridWorld()

# user defined variables
alpha = 0.001
lambda_ = 1
state_space = 24
actions = 4
steps = 25
episodes = 100

# create policy
theta = np.random.uniform(low = 0.0, high=1.0, size=(state_space, actions))
theta = softmax(theta)

td = TD(lambda_, alpha, env, state_space, steps, theta)
td.train(episodes)