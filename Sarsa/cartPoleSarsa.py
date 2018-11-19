# imports
from sarsa import Sarsa
from cartPole import CartPole
import sys
import matplotlib.pyplot as plt
import numpy as np

arg1 = float(sys.argv[1])

# Initializing the gridworld
env = CartPole()

# predefined parameters
gamma = 0.9
alpha = arg1
state_space = 24 #not used in this code
actions = 2
steps = 25 #not used in this code
episodes = 100
e = 0.4
discount=1.0
plot = True
order = 3
trails = 20
rewards = []

for t in range(trails):
    td = Sarsa(gamma, alpha, env, state_space, steps, e, plot=plot, order=order, discount=discount)
    td.train(episodes, trails)
    rewards.append(td.reward)

avg = np.average(np.array(rewards), axis=0)
std = np.std(np.array(rewards), axis=0)
maximumEpisodes = avg.shape[0]
plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
plt.show()

