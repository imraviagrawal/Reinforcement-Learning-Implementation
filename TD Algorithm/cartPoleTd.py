# Imports
import numpy as np
from cartPole import CartPole
from TD_algorithm import TD

# Initializing the cartpole env
env = CartPole()

# user defined variables
alpha = 0.001
lambda_ = 1
state_space = 4
actions = 1
steps = 100
episodes = 100

# create policy
td = TD(lambda_, alpha, env, state_space, steps, order=3)
#td.train(episodes)