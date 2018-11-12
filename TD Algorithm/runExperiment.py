import os
import numpy as np

for i in [0.0001, 0.0005, 0.0009, 0.001, 0.005, 0.009]:
    os.system("python GridWorldTD.py %s %s" %(i, 0.9))
    for j in [3, 5]:
        os.system("python cartPoleTd.py %s %s %s" %(i, j, 1))

os.system("python plotGraph.py")
