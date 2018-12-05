# Imports
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import pickle

# Actor Critic Algorithm class
class Actor_Critic(object):
    # Main class to train the TD algorithm for the n number of episodes
    # the class takes the policy, alpha and lambda as the input
    def __init__(self, gamma, alpha_critic, alpha_actor, env, state_space, steps,  order=3, actions=4, plot=False, discount=0.9, lambda_=0.1):
        self.alpha_critic = alpha_critic
        self.alpha_actor = alpha_actor
        self.gamma = gamma
        self.env = env
        self.state_space = state_space
        self.steps = steps
        self.lambda_ = lambda_
        self.td_error = []
        self.reward = []
        self.order = order
        self.probs = [0.25, 0.25, 0.25, 0.25]
        self.plot = plot
        self.discount = discount
        self.actions = actions
        # self.q_value = np.random.uniform(0, 1, size= (state_space, actions))
        self.w = np.zeros((self.state_space, 1))
        self.theta = np.zeros((self.state_space, self.actions))
        self.eligibility_v = np.zeros((self.state_space, 1))
        self.eligibility_theta = np.zeros((self.state_space, self.actions))
        self.normalization_min = np.array([-1.2, -0.07])
        self.normalization_denominator = np.array([2.4, 0.14])
        if self.env.name != "grid":
            self.c = np.array(list(itertools.product(range(order+1), repeat=self.state_space)))
            self.w = np.zeros((((order + 1) ** self.state_space), 1))
            self.theta = np.zeros((((order + 1) ** self.state_space), self.actions))
            self.eligibility_v = np.zeros((((self.order + 1) ** self.state_space), 1))
            self.eligibility_theta = np.zeros((((self.order + 1) ** self.state_space), self.actions))

    def train(self, episodes):
        # Method to run the actor critic algorithm for n episodes
        # input: episodes, trails
        # return: None
        for _ in range(episodes):
            state = self.env.reset() # reset the environment
            status = self.env.status
            # While we do not reach the terminal state

            # Getting action
            if self.env.name == "cart":
                self.eligibility_v = np.zeros((((self.order + 1) ** self.state_space), 1))
                self.eligibility_theta = np.zeros((((self.order + 1) ** self.state_space), self.actions))

            elif self.env.name == "grid":
                self.eligibility_v = np.zeros((self.state_space, 1))
                self.eligibility_theta = np.zeros((self.state_space, self.actions))

            elif self.env.name == "mountain":
                self.eligibility_v = np.zeros((((self.order + 1) ** self.state_space), 1))
                self.eligibility_theta = np.zeros((((self.order + 1) ** self.state_space), self.actions))

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

                # Getting action # todo make changes as per policy softmax check the implementation
                if self.env.name == "cart":
                    action = self.sampleActionCart(state)

                elif self.env.name == "grid":
                    action, policy= self.sampleActionGrid(state)

                elif self.env.name == "mountain":
                    action, policy = self.sampleActionMountain(state)

                else:
                    assert "Not Supported environment"

                # performing the action in the environment and observing the reward and moving to the new state s_prime
                new_state, reward, status = self.env.performAction(action)
                count += 1
                episode_reward += (self.discount**count)*reward
                # print(episode_reward)
                if status:
                    self.update(reward, state, action)
                    break

                # update the q values according to the previous state and new state
                self.update(reward, state, action, new_state)

                # changing the last state to new state
                state = new_state

            self.reward.append(episode_reward)

    def update(self, reward, s, action, new_s=None, policy=None):
        # Update the value function
        # input: reward, curr_state, and new state
        # return: None (update)

        # Getting the current and next state
        if self.env.name == "grid":
            i, j = s[0], s[1]
            s = i*5+j
            if new_s:
                i_new, j_new = new_s[0], new_s[1]
                new_s= i_new*5 + j_new

            # gettting the last value and new value
            curr_state_value = self.w[s][0]
            next_state_value = self.w[new_s][0]

        else:
            temp_s = np.reshape(np.array(s), (1, self.state_space))
            temp_s = (temp_s - self.normalization_min)/self.normalization_denominator
            phi_s = np.cos(np.dot(self.c, temp_s.T) * math.pi)
            curr_state_value = np.dot(self.w.T, phi_s)[0][0]

            if new_s:
                temp_new_s = np.reshape(np.array(new_s), (1, self.state_space))
                temp_new_s = (temp_new_s - self.normalization_min) / self.normalization_denominator
                phi_new_s = np.cos(np.dot(self.c, temp_new_s.T) * math.pi)
                next_state_value = np.dot(self.w.T, phi_new_s)[0][0]

        # computing the td error
        if new_s:
            delta_t = reward + self.gamma*next_state_value - curr_state_value   # td error
        else:
            delta_t = reward

        # updating the value function if episode is under 100 else calculating
        # the squared error and adding the value to the td_error list.
        if self.env.name == "grid":
            # Critic Update using TD(lambda)
            self.eligibility_v = self.gamma * self.lambda_ * self.eligibility_v
            self.eligibility_v += 1
            self.w = self.w + self.alpha_critic * delta_t * self.eligibility_v

            # actor update
            # delta_pie = np.zeros((self.state_space, self.actions))
            delta_pie = -1*self.theta[s] #todo check this calculation
            delta_pie[action] = 1 - delta_pie[action] # todo check this calculation

            self.eligibility_theta = self.gamma * self.lambda_ * self.eligibility_theta #(state * action)
            self.eligibility_theta[s] += delta_pie
            self.theta = self.theta + self.alpha_actor * delta_t * self.eligibility_theta

        else:
            # Critic Update using TD(lambda)
            self.eligibility_v = self.gamma * self.lambda_ * self.eligibility_v
            self.eligibility_v += phi_s
            self.w = self.w + self.alpha_critic*delta_t*self.eligibility_v

            # actor update
            # delta_pie = np.zeros((((self.order + 1) ** self.state_space), self.actions))
            policy = np.dot(phi_s.T, self.theta)
            delta_pie = -1*np.dot(phi_s, policy.reshape(1, -1)) # check this calculation
            delta_pie[:, action] = phi_s.reshape(-1, ) - delta_pie[:, action] # check this calculation

            self.eligibility_theta = self.gamma * self.lambda_ * self.eligibility_theta
            self.eligibility_theta += delta_pie
            self.theta = self.theta + self.alpha_actor * delta_t * self.eligibility_theta

        self.td_error.append(0)

    #tabular
    def sampleActionGrid(self, state, e_greedy=True):
        i, j = state
        index = i*5+j
        policy = self.softmax(self.theta[index])
        action = np.random.choice(self.env.action, p=policy)
        return action, policy

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
        temp_s = np.reshape(np.array(state), (1, self.state_space))
        temp_s = (temp_s - self.normalization_min) / self.normalization_denominator
        phi_s = np.cos(np.dot(self.c, temp_s.T) * math.pi)
        policy = self.softmax(np.dot(phi_s.T, self.theta))[0]
        action = np.random.choice(self.env.action, p=policy)
        return action, policy

    def softmax(self, x, sigma=1.0):
        x = sigma*x
        mx = np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(x - mx)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        theta_k = numerator / denominator
        return theta_k

    def plotTdError(self):
        plt.plot(self.td_error)
        plt.show()

    def saveTDerror(self):
        if self.env.name == "cart":
            name = "TD_error/" + self.env.name + "_" + str(self.order) + "_" + str(self.alpha) + ".p"
        else:
            name = "TD_error/" + self.env.name + "_" + str(self.alpha) + ".p"
        pickle.dump(self.td_error, open(name , "wb"))


