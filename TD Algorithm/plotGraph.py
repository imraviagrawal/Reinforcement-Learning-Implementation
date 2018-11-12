import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


data = { }
files = [i  for i in os.listdir("./TD_error") if ".p" in i]

r = np.log([0.0001, 0.0005, 0.0009, 0.001, 0.005, 0.009])
for file in files:
    temp = file[:-2]
    temp  = temp.split("_")
    alpha = temp.pop(-1)
    temp = "".join(temp)
    if not temp in data:
        data[temp] = []
    data[temp].append([alpha, np.mean(pickle.load( open( "./TD_error/" + file, "rb" ) ))])

result = []
keys = []
for key in data:
    temp = []
    for curr_value in sorted(data[key]):
        temp.append(curr_value[1])
    result.append(temp)
    keys.append(key)
    plt.plot(r, temp, label= key)
    plt.legend()
    plt.xlabel("Alpha (in logarithmic)")
    plt.ylabel("Mean Squared TD Error")

plt.show()