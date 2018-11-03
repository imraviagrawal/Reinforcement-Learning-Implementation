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

    def train(self):


        reward = 0 # todo
        td_error = None  # pass
        pass