# Imports
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import pickle


# Sarsa Algorithm class
class Sarsa_lambda(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, gamma, alpha, env, state_space, steps, e, order=3, actions=4, plot=False, discount=0.9, lambda_=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.state_space = state_space
        self.q_value = np.random.uniform(0, 1, size= (state_space, actions))
        self.eligibility = np.zeros((state_space, actions))
        self.episolon = e
        self.steps = steps
        self.lambda_ = lambda_
        self.td_error = []
        self.reward = []
        self.order = order
        self.probs = [0.25, 0.25, 0.25, 0.25]
        self.plot = plot
        self.discount = discount
        self.actions = actions
        self.normalization_min = np.array([-1.2, -0.07])
        self.normalization_denominator = np.array([2.4, 0.14])
        if self.env.name != "grid":
            self.c = np.array(list(itertools.product(range(order+1), repeat=self.state_space)))
            self.w = np.zeros((((order + 1) ** self.state_space), self.actions))
            # self.w = np.random.uniform(0, 1, size=(((order + 1) ** self.state_space), self.actions))
            self.eligibility = np.zeros((((order + 1) ** self.state_space), self.actions))
            # self.zeroStack = np.zeros(((order + 1) ** self.state_space)).reshape(((order + 1) ** self.state_space), 1) # 256*1 vector to pad the phi

    def train(self, episodes):

        # Method to run the sarsa algorithm for n episodes
        # input: episodes, trails
        # return: None
        for _ in range(episodes):
            state = self.env.reset() # reset the environment
            status = self.env.status
            # While we do not reach the terminal state

            # Getting action # todo make changes as per policy e-greedy
            if self.env.name == "cart":
                action = self.sampleActionCart(state)

            elif self.env.name == "grid":
                action = self.sampleActionGrid(state)

            elif self.env.name == "mountain":
                action = self.sampleActionMountain(state)

            else:
                assert "Not Supported environment"

            # local variable to store the variable
            count = 0 # count
            episode_reward = 0 # episode reward
            steps = 0
            while not status:
                if steps == self.steps:
                    break
                steps += 1
                # performing the action in the environment and observing the reward and moving to the new state s_prime
                new_state, reward, status = self.env.performAction(action)
                count += 1
                episode_reward += (self.discount**count)*reward
                # print(episode_reward)

                if status:
                    break

                # Choosing the action a_prime at the state s_prime
                if self.env.name == "cart":
                    action_prime = self.sampleActionCart(new_state, e_greedy=False)

                elif self.env.name == "grid":
                    action_prime = self.sampleActionGrid(new_state, e_greedy=False)

                elif self.env.name == "mountain":
                    action_prime = self.sampleActionMountain(state, e_greedy=True)

                else:
                    assert "Not Supported environment"

                # update the q values according to the previous state and new state
                self.update(reward, state, new_state, action, action_prime)

                # changing the last state to new state
                state = new_state
                action = action_prime

            self.reward.append(episode_reward)
            if self.env.name == "mountain":
                self.eligibility = np.zeros((((self.order + 1) ** self.state_space), self.actions))

    def update(self, reward, s, new_s, action, action_prime):
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
            curr_state_value = self.q_value[s, action]
            next_state_value = self.q_value[new_s, action_prime]

            # updating eligibility trace
            self.eligibility = self.gamma * self.lambda_ * self.eligibility
            self.eligibility[s, action] += 1

        else:
            temp_s = np.reshape(np.array(s), (1, self.state_space))
            temp_s = (temp_s - self.normalization_min)/self.normalization_denominator
            temp_new_s = np.reshape(np.array(new_s), (1, self.state_space))
            temp_new_s = (temp_new_s - self.normalization_min) / self.normalization_denominator
            phi_s = np.cos(np.dot(self.c, temp_s.T) * math.pi)

            phi_new_s = np.cos(np.dot(self.c, temp_new_s.T) * math.pi)

            # make changes
            curr_state_value = np.dot(self.w.T, phi_s)[action + 1][0]
            next_state_value = np.dot(self.w.T, phi_new_s)[action_prime][0]

            # updating eligibility trace
            self.eligibility = self.gamma * self.lambda_ * self.eligibility
            self.eligibility[:, action+1] += phi_s.reshape(-1, )


        # computing the td error
        delta_t = reward + self.gamma*next_state_value - curr_state_value   # td error

        # updating the value function if episode is under 100 else calculating
        # the squared error and adding the value to the td_error list.
        if self.env.name == "grid":
            self.q_value = self.q_value + self.alpha*delta_t*self.eligibility

        else:
            self.w = self.w + self.alpha*delta_t*self.eligibility

        self.td_error.append(0)

    #tabular
    def sampleActionGrid(self, state, e_greedy=True):
        i, j = state
        index = i*5+j
        if e_greedy and np.random.rand() < self.episolon:
            action = np.random.choice([0, 1, 2, 3], p=self.probs)
        else:
            action = np.argmax(self.q_value[index, :])
        return action


    def sampleActionCart(self, state, e_greedy=True):
        # if e_greedy
        if e_greedy and np.random.rand() < self.episolon:
            action = np.random.choice([0, 1])

        # linear policy
        else:
            temp_s = np.reshape(np.array(state), (1, self.state_space))
            temp_s = (temp_s - self.normalization_min) / self.normalization_denominator
            phi_s = np.cos(np.dot(self.c, temp_s.T) * math.pi)
            action = 0 if np.dot(self.w.T, np.vstack([self.zeroStack, phi_s]))[0][0] > np.dot(self.w.T, np.vstack([phi_s, self.zeroStack]))[0][0] else 1
        return action

    def sampleActionMountain(self, state, e_greedy=True):
        if e_greedy and np.random.rand() < self.episolon:
            action = np.random.choice(self.env.action, p=self.env.probs)
        else:
            temp_s = np.reshape(np.array(state), (1, self.state_space))
            temp_s = (temp_s - self.normalization_min) / self.normalization_denominator
            phi_s = np.cos(np.dot(self.c, temp_s.T) * math.pi)
            action = np.argmax(np.dot(phi_s.T, self.w)[0]) - 1
        return action

    def plotTdError(self):
        plt.plot(self.td_error)
        plt.show()

    def saveTDerror(self):
        if self.env.name == "cart":
            name = "TD_error/" + self.env.name + "_" + str(self.order) + "_" + str(self.alpha) + ".p"
        else:
            name = "TD_error/" + self.env.name + "_" + str(self.alpha) + ".p"
        pickle.dump(self.td_error, open(name , "wb"))


