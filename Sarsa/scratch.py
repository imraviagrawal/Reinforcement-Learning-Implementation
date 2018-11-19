import numpy as np

weights = np.random.uniform(size=(4,2))

observation = np.array([0.1, 0.2, 0.3, 0.4]).reshape(1, 4)

print(observation)
action = 1

test = np.zeros((4, 2))
test[:, action] += observation.reshape(4,)
print(test)