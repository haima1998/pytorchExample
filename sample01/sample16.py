##1.2 demo concatenate

import numpy as np
from numpy import random as nr

a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

print(a)
print(b)

c = np.concatenate((a, b), axis=0)
print(c)
d = np.concatenate((a, b.T), axis=1)
print(d)