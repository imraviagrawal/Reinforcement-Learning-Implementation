# Imports
import numpy as np
import math

# Class for cartpole
# when we analize the class the we will define the environment with all the
# predefined variables
class CartPole():
    # there will be two action 1 and -1
    # and force will be have sign accordingly.
    def __init__(self):
        self.F = 10.0
        self.g = 9.8
        self.l = 0.5
        self.tau = 0.02
        self.bound = 3.0
        self.mc = 1.0
        self.mp = 0.1
        self.total_mass = self.mc + self.mp

        self.FailAngle = math.pi/2
        self.state = self.state = np.random.uniform(low=-0.03, high=0.03, size=(4,)) # x, Dx, theta, Dtheta # Initial state
        self.status = False


    def performAction(self, action):
        curr_state = self.state
        x, x_dot, theta, theta_dot = curr_state
        f = self.F if action==1 else -self.F

        # calculating theta
        temp  = (-f-self.mp*self.l*theta_dot*theta_dot*math.sin(theta))/self.total_mass
        theta_denom = self.l*((4.0/3.0) - ((self.mp* math.cos(theta)*math.cos(theta))/(self.total_mass)))
        theta_2dot = (self.g*math.sin(theta) + math.cos(theta)*temp)/(theta_denom)

        # Calculating x_2dot
        x_2dot = (f + self.mp*self.l*(theta_dot*theta_dot*math.sin(theta) - theta_2dot*math.cos(theta)))/(self.total_mass)

        # Updating x, x_dot, theta and theta_dot
        x = x + self.tau*x_dot
        x_dot = x_dot + self.tau*x_2dot
        theta = theta + self.tau*theta_dot
        theta_dot = theta_dot + self.tau*theta_2dot

        # Updating the state
        self.state = (x, x_dot, theta, theta_dot)
        self.status = (abs(x) > self.bound) or (abs(theta) > self.FailAngle)
        rewards = 1

        return self.state, rewards, self.status


    def reset(self):
        self.state = np.random.uniform(low=-0.03, high=0.03, size=(4,))
        self.status = False
        return self.state
