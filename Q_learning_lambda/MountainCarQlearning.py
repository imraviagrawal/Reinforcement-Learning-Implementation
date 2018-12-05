# imports
from policy.q_learning import Q_learning
from environment.MountainCar import mountaincar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

# Initializing the gridworld
env = mountaincar()

# predefined parameters
gamma = 1.0
state_space = 2
actions = 3
steps = 1000
episodes = 100
# e = 0.5
plot = True
discount=1.0
trails = 5
rewards = []

# best parameter, order 5, e 0.2, alpha 0.5
for order in [2]:
    for e in [0.01]:
        for alpha in [0.01]:
            for lambda_ in [0.0]:
                rewards = []
                print("Alpha: ", alpha)
                for t in tqdm(range(trails)):
                    td = Q_learning(gamma, alpha, env, state_space, steps, e, plot=plot, discount=discount,lambda_=lambda_, actions=actions, order=order)
                    td.train(episodes)
                    rewards.append(td.reward)

                avg = np.average(np.array(rewards), axis=0)
                std = np.std(np.array(rewards), axis=0)
                maximumEpisodes = avg.shape[0]
                plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
                name = "return/MountainQ_alpha%s_e%s_lambda%s_order%s.jpg" %(alpha, e, lambda_, order)
                # pickle.dump(avg, open(name, "wb"))
                plt.xlabel("Number of episodes")
                plt.ylabel("Total Reward")
                plt.show()
                # plt.savefig(name)
                # plt.close()