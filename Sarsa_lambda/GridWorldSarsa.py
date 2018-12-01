# imports
from sarsa_lambda import Sarsa_lambda
from GridWorld import gridWorld
# import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

# Initializing the gridworld
env = gridWorld()

# predefined parameters
gamma = 0.9
state_space = 24
actions = 4
steps = 25
episodes = 100
e = 0.3
discount=1.0
plot = True
trails = 100
# lambda_ = 0.1
# rewards = []

type = "grid"
# best parameter, order 3, e 0.2, alpha 0.5
# best parameter, order 5, e 0.2, alpha 0.5
for e in [1.0]:#, 0.01, 0.3, 0.4]:
    for alpha in [0.1]:#0.001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1, 0.5]:
        for lambda_ in [0.3]:
            rewards = []
            print("Alpha: ", alpha)
            for t in tqdm(range(trails)):
                # print("Alpha: %s, Trail: %s" %(alpha, t))
                td = Sarsa_lambda(gamma, alpha, env, state_space, steps, e, plot=plot, discount=discount, lambda_=lambda_)
                td.train(episodes)
                rewards.append(td.reward)

            avg = np.average(np.array(rewards), axis=0)
            std = np.std(np.array(rewards), axis=0)
            maximumEpisodes = avg.shape[0]
            plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
            name = "Grid_alpha%s_e%s.jpg" %(alpha, e)
            pickle.dump(avg, open(name, "wb"))
            plt.xlabel("Number of episodes")
            plt.ylabel("Total Reward")
            plt.show()