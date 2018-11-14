# Imports

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import pickle


# Sarsa Algorithm class
class Sarsa(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, gamma, alpha, env, state_space, steps, e, order=3, actions=4):
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.state_space = state_space
        self.q_value = np.random.uniform(size= (state_space, actions))
        self.episolon = e
        self.steps = steps
        self.td_error = []
        self.order = order
        self.probs = [0.25, 0.25, 0.25, 0.25]
        if self.env.name == "cart":
            self.c = np.array(list(itertools.product(range(order+1), repeat=4)))
            self.w = np.zeros(((order+1)**4)).reshape(((order+1)**4), 1)


    def train(self, episodes):
        # Method to run the sarsa algorithm for n episodes
        # input: episodes
        # return: None
        for _ in range(episodes):
            status = False
            state = self.env.reset() # reset the environment

            # While we do not reach the terminal state

            # Getting action # todo make changes as per policy e-greedy
            if self.env.name == "cart":
                action = self.sampleActionCart(state) # todo episolon greedy policy

            elif self.env.name == "grid":
                action = self.sampleActionGrid(state) # todo episolon greedy policy

            else:
                assert "Not Supported environment"

            while not status:

                # performing the action in the environment and observing the reward and moving to the new state s_prime
                new_state, reward, status = self.env.performAction(action)

                # Choosing the action a_prime at the state s_prime
                if self.env.name == "cart":
                    action_prime = self.sampleActionCart(new_state)  # todo sarsa policy change action function to accomdate the q value

                elif self.env.name == "grid":
                    action_prime = self.sampleActionGrid(new_state) # todo sarsa policy

                else:
                    assert "Not Supported environment"

                # update the q values according to the previous state and new state
                self.update(reward, state, new_state)

                # changing the last state to new state
                state = new_state
                action = action_prime


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

            # gettting the last value and new value
            curr_state_value = self.q_value[s]
            next_state_value = self.q_value[new_s]

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

        if self.env.name == "grid":
            self.q_value[s] = self.q_value[s] + self.alpha*delta_t

        else:
            self.w = self.w + self.alpha*np.multiply(np.array(delta_t), phi_s)

    # softmax tabular
    def sampleActionGrid(self, state, e_greedy=False):
        i, j = state
        index = i*5+j
        if e_greedy:
            action = np.random.choice([0, 1, 2, 3], p=self.probs)
        else:
            action = np.argmax(self.q_value[index, :])
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


