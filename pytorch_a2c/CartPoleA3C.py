# imports
import gym

import math
import random
import numpy as np

import matplotlib.pyplot as plt

from policy.actor_critic import ActorCritic
from policy.multiprocessing_env import SubprocVecEnv

import torch
import torch.optim as optim

# Helper function
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward

    if vis:
        print(total_reward)
        env.close()
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma*R*masks[step]
        returns.insert(0, R)
    return returns

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

# Creating Environment
num_envs = 16
env_name = "CartPole-v0"

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)
env  = gym.make(env_name)

# input output dimensions
num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

# hyper parameters
hidden_size = 256
lr          = 3e-4
num_steps   = 5

max_frames   = 20000
frame_idx    = 0
test_rewards = []

# Setting the GPU settings
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Model
model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

state = envs.reset()

while frame_idx < max_frames:

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    for i in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        state = next_state
        frame_idx += 1

        if frame_idx % 1000 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_env(True)



