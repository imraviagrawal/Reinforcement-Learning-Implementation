# imports
from q_learning import Q_learning
from GridWorld import gridWorld
import sys

arg1 = float(sys.argv[1])

# Initializing the gridworld
env = gridWorld()

# predefined parameters
gamma = 0.9
alpha = arg1
state_space = 24
actions = 4
steps = 25
episodes = 100
e = 0.2
plot = True

td = Q_learning(gamma, alpha, env, state_space, steps,e,  plot=plot)
td.train(episodes)