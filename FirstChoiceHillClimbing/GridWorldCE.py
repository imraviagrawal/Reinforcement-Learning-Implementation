# Imports
import numpy as np
from GridWorld import gridWorld
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import pickle
import time, sys

name = sys.argv[1]
# Predefined variables
K = 20 # Trails
Ke = 3 # elite population
e = 100 # Episolon
maximumEpisodes = 50 # Total number of times running the policy for KN times
N = 5 # number of times to run the policy
steps = 20 # to travel in the policy
state_space = 24
sigma = 2.25
actions = 4
trails = 500
numOfThreads = 1
gamma = 1.0

# initialing the gridWorld environment
env = gridWorld()

# Initializing the theta and cov
all_rewards = []

# Function to perform sigmoid calculation
def sigmoid(theta, sigma= 1.0):
    x = sigma * theta
    x = x.reshape(state_space, actions)
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    theta_k = numerator / denominator
    return theta_k

# softmax tabular
def sampleAction(state, s):
    i, j = state
    index = i*5 + j
    probs = s[index]
    action = np.random.choice([0, 1, 2, 3], p = probs)
    return action

# Evaluate the policy
def evaluate(state, theta, steps):
    curr_reward = 0
    for episode in range(steps):
        action = sampleAction(state, theta)
        state, reward, status = env.performAction(action)
        curr_reward += (gamma**episode)*reward
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
            s = sigmoid(theta_k, sigma)
            curr_reward = 0

            for n in range(N):
                state = env.reset()
                curr_reward += evaluate(state, s, steps)  # Evaluating policy

            total_rewards.append(curr_reward / N)
            all_theta.append(theta_k)  # Appending the policy theta_k


        # best ke policy with top rewards
        indices = np.argsort(np.array(total_rewards))[::-1]
        indices = indices[:Ke]

        # print(total_rewards)
        # Selecting policy with top rewards
        all_theta = np.array(all_theta)
        theta_ke = all_theta[indices]

        # Calculating the theta
        theta = np.sum(theta_ke, axis=0) / Ke

        # calculating theta_k - theta
        temp = np.zeros((actions*state_space, actions*state_space))
        # print(temp)
        # temp = np.identity(state_space * actions)
        for i in range(Ke):
            theta_theta_ke = np.array(theta_ke[i] - theta)
            temp += np.dot(theta_theta_ke.T, theta_theta_ke)

            #temp += np.outer(np.array(theta_ke[i] - theta), np.array(theta_ke[i] - theta).T)

        # calculating cov
        cov = (e * np.identity(state_space * actions) + temp) / (e + Ke)

        # Adding avg and std of reward for last policy
        avg_rewards += total_rewards

    print("Policy completed %s" %t)
    return avg_rewards

# Multiple Processing to speed up the process
# pool = ThreadPool(numOfThreads)
start = time.time()
for t in range(trails):
    print("Trials: ", t)
    theta = np.zeros(state_space*actions)
    cov = np.identity(state_space*actions)
    all_rewards.append(CEM(theta, cov, start, t))


print(len(all_rewards))
print(time.time() - start)
result = {"result":all_rewards}
file_name = "gridWorldCEResults/save%s.p" %name
pickle.dump(result, open(file_name, "wb" ))
# all_rewards = np.average(np.array(all_rewards), axis=0)
# plt.plot(all_rewards)
# # #plt.errorbar(np.array([i for i in range(maximumEpisodes)]).reshape(1, -1), np.array(avg_rewards).reshape(1, -1), std_rewards, marker='^', ecolor='g')
# plt.show()

avg = np.average(np.array(all_rewards), axis=0)
std = np.std(np.array(all_rewards), axis=0)
maximumEpisodes = avg.shape[0]
# plt.plot(avg)
plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
plt.show()



