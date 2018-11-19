# imports
from sarsa import Sarsa
from cartPole import CartPole
import sys

arg1 = float(sys.argv[1])

# Initializing the gridworld
env = CartPole()

# predefined parameters
gamma = 0.9
alpha = arg1
state_space = 24
actions = 4
steps = 25
episodes = 500
e = 0.2
plot = True

td = Sarsa(gamma, alpha, env, state_space, steps, e, plot=plot)
td.train(episodes)