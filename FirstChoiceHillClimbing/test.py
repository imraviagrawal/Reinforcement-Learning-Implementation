import pickle
import matplotlib.pyplot as plt
import numpy as np

# all_rewards = pickle.load( open( "./gridWorldCEResults/save0.p", "rb" ) )
# all_rewards = pickle.load( open( "./gridWorldCEResults/save5.p", "rb" ) )
# all_rewards = pickle.load( open( "./gridWorldCEResults/cartpolefchcsave0.p", "rb" ) )
all_rewards = pickle.load( open( "./gridWorldCEResults/cartpolesave0.p", "rb" ) )
# all_rewards = pickle.load( open( "./gridWorldCEResults/GridWorldfchcsave0.p", "rb" ) )
all_rewards = np.array(all_rewards["result"])
avg = np.average(all_rewards, axis=0)
std = np.std(all_rewards, axis=0)
maximumEpisodes = avg.shape[0]

plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
plt.ylabel("Reward")
plt.xlabel("Episodes")
plt.show()
