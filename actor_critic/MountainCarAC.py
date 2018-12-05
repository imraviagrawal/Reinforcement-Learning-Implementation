# imports
from policy.actor_critic import Actor_Critic
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
discount=1.0
plot = True
trails = 5
# order = 3

# best parameter, order 3, e 0.2, alpha 0.5
# best parameter, order 5, e 0.2, alpha 0.5
for order in [1]:
    for e in [0.00]:
        for alpha_critic in [0.001]:
            for alpha_actor in [0.001]:
                for lambda_ in [0.65]:
                    rewards = []
                    print("Alpha: ", (alpha_critic, alpha_actor))
                    for t in tqdm(range(trails)):
                        # print("Alpha: %s, Trail: %s" %(alpha, t))
                        td = Actor_Critic(gamma=gamma, env=env, state_space=state_space,steps= steps, plot=plot, discount=discount, lambda_=lambda_, order=order, actions=actions, alpha_critic=alpha_critic, alpha_actor=alpha_actor)
                        td.train(episodes)
                        rewards.append(td.reward)

                    avg = np.average(np.array(rewards), axis=0)
                    std = np.std(np.array(rewards), axis=0)
                    maximumEpisodes = avg.shape[0]
                    name = "return/mountain_sarsa_zero/Mountain_alphas%s_%s_e%s_lambda%s_order%s.jpg" %(alpha_critic, alpha_actor, e, lambda_, order)
                    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
                    # pickle.dump(avg, open(name, "wb"))
                    plt.xlabel("Number of episodes")
                    plt.ylabel("Total Reward")
                    plt.show()
                    # plt.savefig(name)
                    plt.close()