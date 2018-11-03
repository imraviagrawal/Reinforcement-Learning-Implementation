# imports
import numpy as np
from cartPole import CartPole
import matplotlib.pyplot as plt
import pickle
import time, sys

name = sys.argv[1]

# Predefined variables
maximumEpisodes = 100 # Total number of times running the policy for KN times
N = 10 # number of times to run the policy
state_space = 4 # state space
sigma = 1
trail = 500
actions = 1
steps = 50 # to travel in the policy

# initialing the cart pole environment
env = CartPole()

# Sample Action
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

def fchc(theta, total_rewards, bestReward, start):
    # While converges
    for i in range(maximumEpisodes):
        print(i, time.time() - start)
        # Sampling the policy
        theta_d = np.random.multivariate_normal(theta, sigma * np.identity(state_space))
        currReward = 0

        # Evaluating the new policy N times
        for n in range(N):
            state = env.reset()
            currReward += evaluate(state, theta_d, steps)


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
    currReward = 0
    total_rewards = []

    # Evaluating policy N times
    for n in range(N):
        state = env.reset()
        currReward += evaluate(state, theta, steps)

    bestReward = currReward/N
    total_rewards.append(bestReward)
    all_rewards.append(fchc(theta, total_rewards, bestReward, start))



print(len(all_rewards))
print(time.time() - start)
result = {"result":all_rewards}
file_name = "gridWorldCEResults/cartpolefchcsave%s.p" %name
pickle.dump(result, open(file_name, "wb" ))
all_rewards = np.average(np.array(all_rewards), axis=0)
plt.plot(all_rewards)
# #plt.errorbar(np.array([i for i in range(maximumEpisodes)]).reshape(1, -1), np.array(avg_rewards).reshape(1, -1), std_rewards, marker='^', ecolor='g')
plt.show()
