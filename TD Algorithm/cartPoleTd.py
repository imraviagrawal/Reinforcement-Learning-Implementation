# Imports
import numpy as np
from cartPole import CartPole
from TD_algorithm import TD
import sys
# Initializing the cartpole env
env = CartPole()

arg = float(sys.argv[1])

# user defined variables
alpha = arg
lambda_ = 1
state_space = 4
actions = 1
steps = 100
episodes = 100
order = 3
# create policy
td = TD(lambda_, alpha, env, state_space, steps, order=order)
td.train(episodes)