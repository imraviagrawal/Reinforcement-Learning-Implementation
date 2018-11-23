# Imports
import numpy as np
from cartPole import CartPole
import matplotlib.pyplot as plt
import pickle
import time, sys

name = sys.argv[1]
# Predefined variables
K = 20 # Trails
Ke = 4 # elite population
e = 0.0001 # Episolon
maximumEpisodes = 50 # Total number of times running the policy for KN times
N = 10 # number of times to run the policy
steps =  500 # to travel in the policy
state_space = 4
sigma = 1
actions = 1
trails = 3
numOfThreads = 1

# initialing the gridWorld environment
env = CartPole()

# Initializing the theta and cov
all_rewards = []

# softmax tabular
def sampleAction(state, theta):
    if np.dot(state, theta.T) > 0:
        return 1
    return 0

# Evaluate the policy
def evaluate(state, theta, steps):
    curr_reward = 0
    for episode in range(steps):
        action = sampleAction(state, theta)
        state, reward, status = env.performAction(action)
        curr_reward += reward
        if status:
            break
    return curr_reward

# Function to calculate cross entropy
def CEM(theta, cov, start, t):
    avg_rewards = []
    for i in range(maximumEpisodes):  # While or some large loop
        print(i, time.time() - start)
        # reseting the environment
        total_rewards = []
        all_theta = []

        # sampling the theta for K times
        for k in range(K):
            # Normal sampling theta
            theta_k = np.random.multivariate_normal(theta, cov)  # Sampling theta_k
            curr_reward = 0

            #pool1 = ThreadPool(N)
            # Evaluating the policy for K times
            #curr_reward = []
            for n in range(N):
                state = env.reset()
                curr_reward += evaluate(state, theta_k, steps)  # Evaluating policy

            total_rewards.append(curr_reward / N)
            all_theta.append(theta_k)  # Appending the policy theta_k


        # best ke policy with top rewards
        indices = np.argsort(np.array(total_rewards))[::-1]
        indices = indices[:Ke]


        # Selecting policy with top rewards
        all_theta = np.array(all_theta)
        theta_ke = all_theta[indices]

        # Calculating the theta
        theta = np.sum(theta_ke, axis=0) / Ke

        # calculating theta_k - theta
        temp = 0
        # temp = np.identity(state_space * actions)
        for i in range(Ke):
            temp += np.outer(np.array(theta_ke[i] - theta), np.array(theta_ke[i] - theta).T)

        # calculating cov
        cov = (e * np.identity(state_space * actions) + temp) / (e + Ke)

        # Adding avg and std of reward for last policy
        avg_rewards += total_rewards

    print("Policy completed %s" %t)
    return avg_rewards

# Multiple Processing to speed up the process
start = time.time()
for t in range(trails):
    print("Trials: ", t)
    theta = np.zeros(state_space*actions)
    cov = np.identity(state_space*actions)
    all_rewards.append(CEM(theta, cov, start, t))


print(len(all_rewards))
print(time.time() - start)
result = {"result":all_rewards}
file_name = "gridWorldCEResults/cartpolesave%s.p" %name
pickle.dump(result, open(file_name, "wb" ))
# print(len(all_rewards))
# all_rewards = np.average(np.array(all_rewards), axis=0)
# plt.plot(all_rewards)
# plt.errorbar(np.array([i for i in range(maximumEpisodes)]).reshape(1, -1), np.array(avg_rewards).reshape(1, -1), std_rewards, marker='^', ecolor='g')
# plt.show()


