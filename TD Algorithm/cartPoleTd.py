# Imports
import numpy as np
from cartPole import CartPole
from TD_algorithm import TD
import sys
# Initializing the cartpole env
env = CartPole()

arg1 = float(sys.argv[1])
arg2 = int(sys.argv[2])
arg3 = float(sys.argv[3])

# user defined variables
alpha = arg1
order = arg2
lambda_ = arg3
state_space = 4
actions = 1
steps = 100
episodes = 100


# create policy
td = TD(lambda_, alpha, env, state_space, steps, order=order)
td.train(episodes)