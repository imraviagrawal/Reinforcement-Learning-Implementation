# imports
import numpy as np


def softmax(x, sigma= 1.0):
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    theta_k = numerator / denominator
    return theta_k