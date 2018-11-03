# Imports

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time, sys


# TD Algorithm class
class TD(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, lambda_, alpha, env):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.env = env

    def train(self, episodes):
        # Method to run the td algorithm for n episodes
        # input: episodes
        # return: None
        for _ in range(episodes):
            state = self.env.reset() # reset the environment
            while not self.env.status:

                # Getting action
                action = 0 #todo

                # performing the action in the environment
                new_state, reward, status = self.env.performAction(action)

                # todo update
                state = new_state


    def update(self, reward, state, new_state):
        # Getting the current and next state
        curr_state = 0 #todo
        new_state = 0  #todo

        # computing the td error
        delta_t = reward + self.lambda_*next_state - curr_state   # td error


        # updating the 



