# imports
from policy.actor_critic import Actor_Critic
from environment.GridWorld import gridWorld
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
steps = 50
episodes = 150
discount=1.0
plot = True
trails = 20

type = "grid"
# best parameter, order 3, e 0.2, alpha 0.5
# best parameter, order 5, e 0.2, alpha 0.5

for alpha_critic in [0.001]:
    for alpha_actor in [0.1, 0.20]:
        for lambda_ in [0.8, 0.9]:
            rewards = []
            print("Alpha: ", (alpha_critic, alpha_actor))
            for t in tqdm(range(trails)):
                # print("Alpha: %s, Trail: %s" %(alpha, t))
                td = Actor_Critic(gamma=gamma, env=env, state_space=state_space, steps=steps, plot=plot,
                                  discount=discount, lambda_=lambda_, actions=actions,
                                  alpha_critic=alpha_critic, alpha_actor=alpha_actor)
                td.train(episodes)
                rewards.append(td.reward)

            avg = np.average(np.array(rewards), axis=0)
            std = np.std(np.array(rewards), axis=0)
            maximumEpisodes = avg.shape[0]
            plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
            name = "return/grid/Grid_alpha%s_%s_lambda_%s.jpg" %(alpha_critic, alpha_actor, lambda_)
            # pickle.dump(avg, open(name, "wb"))
            plt.xlabel("Number of episodes")
            plt.ylabel("Total Reward")
            plt.savefig(name)
            plt.close()
            # plt.show()