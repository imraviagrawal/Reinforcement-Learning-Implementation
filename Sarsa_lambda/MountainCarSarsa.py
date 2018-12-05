# imports
from policy.sarsa_lambda import Sarsa_lambda
from environment.MountainCar import mountaincar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

# Initializing the gridworld
env = mountaincar()

# predefined parameters
gamma = 0.9
state_space = 2
actions = 3
steps = 1000
episodes = 100
discount=1.0
plot = True
trails = 100
# order = 3

# best parameter, order 3, e 0.2, alpha 0.5
# best parameter, order 5, e 0.2, alpha 0.5
for order in [5, 7]:
    for e in [0.01]:
        for alpha in [0.001]:
            for lambda_ in [0.0]:
                rewards = []
                print("Alpha: ", alpha)
                for t in tqdm(range(trails)):
                    # print("Alpha: %s, Trail: %s" %(alpha, t))
                    td = Sarsa_lambda(gamma, alpha, env, state_space, steps, e, plot=plot, discount=discount, lambda_=lambda_, order=order, actions=actions)
                    td.train(episodes)
                    rewards.append(td.reward)

                avg = np.average(np.array(rewards), axis=0)
                std = np.std(np.array(rewards), axis=0)
                maximumEpisodes = avg.shape[0]
                name = "return/mountain_sarsa_zero/Mountain_alpha%s_e%s_lambda%s_order%s.jpg" %(alpha, e, lambda_, order)
                plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
                # pickle.dump(avg, open(name, "wb"))
                plt.xlabel("Number of episodes")
                plt.ylabel("Total Reward")
                # plt.show()
                plt.savefig(name)
                plt.close()