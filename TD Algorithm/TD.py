# Imports

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time, sys


# TD Algorithm class
class TD(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, lambda_, alpha, env, state_space, steps, theta):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.env = env
        self.state_space = state_space
        self.value_function = np.zeros(state_space)
        self.steps = steps
        self.theta = theta

    def train(self, episodes):
        # Method to run the td algorithm for n episodes
        # input: episodes
        # return: None
        for _ in range(episodes):
            state = self.env.reset() # reset the environment
            for _ in range(self.steps):

                # Getting action
                if self.env.name == "cart":
                    action = 0 #todo

                elif self.env.name == "grid":
                    action = sampleActionGrid(state, self.theta) #todo

                else:
                    assert "Not Supported environment"

                # performing the action in the environment
                new_state, reward, status = self.env.performAction(action)

                # update value function
                self.update(reward, state, new_state)


                state = new_state

                if status:
                    break

    def update(self, reward, s, new_s):
        # Update the value function
        # input: reward, curr_state, and new state
        # return: None (update)

        # Getting the current and next state
        if self.env.name == "grid":
            i, j = s[0], s[1]
            i_new, j_new = new_s[0], new_s[1]
            s = i*5+j
            new_s= i_new*5 + j_new

        curr_state_value = self.value_function[s]
        next_state_value = self.value_function[new_s]

        # computing the td error
        delta_t = reward + self.lambda_*next_state_value - curr_state_value   # td error


        # updating the value function
        self.value_function[s] = self.value_function[s] + self.alpha*delta_t


    # softmax tabular
    def sampleActionGrid(state, s):
        i, j = state
        index = i * 5 + j
        probs = s[index]
        action = np.random.choice([0, 1, 2, 3], p=probs)
        return action


