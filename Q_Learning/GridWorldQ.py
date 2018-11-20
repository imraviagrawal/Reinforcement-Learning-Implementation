# imports
from q_learning import Q_learning
from GridWorld import gridWorld
import sys
import matplotlib.pyplot as plt
import numpy as np

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
e = 0.3
plot = True
discount=1.0
trails = 100
rewards = []

# Best parameters e 0.3, gamma 0.9, alpha 0.5

for t in range(trails):
    print("Trail: ", t)
    td = Q_learning(gamma, alpha, env, state_space, steps, e,  plot=plot, discount=discount)
    td.train(episodes)
    rewards.append(td.reward)

avg = np.average(np.array(rewards), axis=0)
std = np.std(np.array(rewards), axis=0)
maximumEpisodes = avg.shape[0]
plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
plt.show()