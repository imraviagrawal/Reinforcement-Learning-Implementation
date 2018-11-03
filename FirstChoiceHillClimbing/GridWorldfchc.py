# imports
import numpy as np
from GridWorld import gridWorld
import matplotlib.pyplot as plt
import pickle
import time, sys

name = sys.argv[1]

# Predefined variables
maximumEpisodes = 150 # Total number of times running the policy for KN times
N = 50 # number of times to run the policy
state_space = 24 # state space
sigma = 1
trail = 500
actions = 4
steps = 15 # to travel in the policy

# initialing the cart pole environment
env = gridWorld()

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
        curr_reward += reward
        if status:
            break
    return curr_reward

def fchc(theta, total_rewards, bestReward, start):
    # While converges
    for i in range(maximumEpisodes):
        print(i, time.time() - start)
        # Sampling the policy
        theta_d = np.random.multivariate_normal(theta, sigma * np.identity(state_space*actions))
        s = sigmoid(theta_d, sigma)
        currReward = 0

        # Evaluating the new policy N times
        for n in range(N):
            state = env.reset()
            currReward += evaluate(state, s, steps)


        currReward = currReward/N
        total_rewards.append(currReward)

        # Updating the policy if the curr policy is better than last policy
        if currReward >= bestReward:
            theta = theta_d
            bestReward = currReward

    return total_rewards



all_rewards = []
bestReward = []
start = time.time()
for t in range(trail):
    print("Trials: ", t)

    theta = np.zeros(actions*state_space)
    s = sigmoid(theta, 1.0)
    currReward = 0
    total_rewards = []

    # Evaluating policy N times
    for n in range(N):
        state = env.reset()
        currReward += evaluate(state, s, steps)

    bestReward = currReward/N
    total_rewards.append(bestReward)
    all_rewards.append(fchc(theta, total_rewards, bestReward, start))



print(len(all_rewards))
print(time.time() - start)
result = {"result":all_rewards}
file_name = "gridWorldCEResults/GridWorldfchcsave%s.p" %name
pickle.dump(result, open(file_name, "wb" ))
# all_rewards = np.average(np.array(all_rewards), axis=0)
# plt.plot(all_rewards)
# # #plt.errorbar(np.array([i for i in range(maximumEpisodes)]).reshape(1, -1), np.array(avg_rewards).reshape(1, -1), std_rewards, marker='^', ecolor='g')
# plt.show()

avg = np.average(np.array(all_rewards), axis=0)
std = np.std(np.array(all_rewards), axis=0)
maximumEpisodes = avg.shape[0]

plt.errorbar(np.array([i for i in range(maximumEpisodes)]), avg, std, marker='^', ecolor='g')
plt.show()
