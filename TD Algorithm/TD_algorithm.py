# Imports

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import pickle


# TD Algorithm class
class TD(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, lambda_, alpha, env, state_space, steps, order=3):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.env = env
        self.state_space = state_space
        self.value_function = np.zeros(state_space)
        self.steps = steps
        self.td_error = []
        self.order = order
        self.probs = [0.25, 0.25, 0.25, 0.25]
        if self.env.name == "cart":
            self.c = np.array(list(itertools.product(range(order+1), repeat=4)))
            self.w = np.zeros(((order+1)**4)).reshape(((order+1)**4), 1)


    def train(self, episodes):
        # Method to run the td algorithm for n episodes
        # input: episodes
        # return: None
        for _ in range(episodes + 100):
            state = self.env.reset() # reset the environment
            for s in range(self.steps):

                # Getting action
                if self.env.name == "cart":
                    action = self.sampleActionCart(state)

                elif self.env.name == "grid":
                    action = self.sampleActionGrid(state)

                else:
                    assert "Not Supported environment"

                # performing the action in the environment
                new_state, reward, status = self.env.performAction(action)

                if status:
                    break
                # update value function
                if _ < 100:
                    self.update(reward, state, new_state)
                else:
                    self.update(reward, state, new_state, True)

                state = new_state

        # self.plotTdError()
        self.saveTDerror()


    def update(self, reward, s, new_s, squared_error=False):
        # Update the value function
        # input: reward, curr_state, and new state
        # return: None (update)

        # Getting the current and next state
        if self.env.name == "grid":
            i, j = s[0], s[1]
            i_new, j_new = new_s[0], new_s[1]
            s = i*5+j
            new_s= i_new*5 + j_new

            # gettting the last value and new value
            curr_state_value = self.value_function[s]
            next_state_value = self.value_function[new_s]

        else:
            temp_s = np.reshape(np.array(s), (1, 4))
            temp_new_s = np.reshape(np.array(new_s), (1, 4))
            phi_s = np.cos(np.dot(self.c, temp_s.T)*math.pi)
            phi_new_s = np.cos(np.dot(self.c, temp_new_s.T) * math.pi)
            curr_state_value = np.dot(self.w.T, phi_s)[0]
            next_state_value = np.dot(self.w.T, phi_new_s)[0]

        # computing the td error
        delta_t = reward + self.lambda_*next_state_value - curr_state_value   # td error
        # updating the value function if episode is under 100 else calculating
        # the squared error and adding the value to the td_error list.
        if not squared_error:
            if self.env.name == "grid":
                self.value_function[s] = self.value_function[s] + self.alpha*delta_t

            else:
                self.w = self.w + self.alpha*np.multiply(np.array(delta_t), phi_s)

        else:
            self.td_error.append(delta_t*delta_t)

    # softmax tabular
    def sampleActionGrid(self, state):
        action = np.random.choice([0, 1, 2, 3], p=self.probs)
        return action


    def sampleActionCart(self, state):
        probs = np.random.uniform()
        if probs > 0.5:
            return 1
        return 0

    def plotTdError(self):
        plt.plot(self.td_error)
        plt.show()

    def saveTDerror(self):
        if self.env.name == "cart":
            name = "TD_error/" + self.env.name + "_" + str(self.order) + "_" + str(self.alpha) + ".p"
        else:
            name = "TD_error/" + self.env.name + "_" + str(self.alpha) + ".p"
        pickle.dump(self.td_error, open(name , "wb"))


