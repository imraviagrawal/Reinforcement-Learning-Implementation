"""File to run the gridWorld algorithm
   run instructions needs to be updated
"""

# imports
from TD_algorithm import TD
from GridWorld import gridWorld
import sys

arg1 = float(sys.argv[1])
arg2 = float(sys.argv[2])

# Initializing the gridworld
env = gridWorld()

# user defined variables
alpha = arg1
lambda_ = arg2
state_space = 24
actions = 4
steps = 25
episodes = 100

td = TD(lambda_, alpha, env, state_space, steps)
td.train(episodes)