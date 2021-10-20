##1.2 demo stack append with a new dim

import numpy as np
from numpy import random as nr

a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a)
print(b)
print(np.stack((a, b), axis=0))
print(a.shape)
print(b.shape)
print(np.stack((a, b), axis=0).shape)