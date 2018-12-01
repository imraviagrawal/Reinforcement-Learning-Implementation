import pickle
import matplotlib.pyplot as plt

all_rewards1 = pickle.load( open( "Grid_alpha0.01_e0.3.jpg", "rb" ) )
all_rewards2 = pickle.load( open( "../Q_learning/Grid_alpha0.001_e0.1.jpg", "rb" ) )
plt.plot(all_rewards1, label='Cart Pole Sarsa')
plt.plot(all_rewards2, label='Cart Pole Q Learning')
plt.xlabel("Number of episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.show()

# file2_reward = pickle.load()
