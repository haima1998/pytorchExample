##1.2 demo append

import numpy as np
from numpy import random as nr

a =np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)

print(a)
print(b)
print(c) 

print('..................')
a =np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
print(a)
print(b)
# 按行合并
c = np.append(a, b, axis=0)
print('append by row')
print(c)
print('c.shape:', c.shape)
# 按列合并
d = np.append(a, b, axis=1)
print('append by cols')
print(d)
print('d.shape', d.shape)