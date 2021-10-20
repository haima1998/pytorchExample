##1.2 demo ravel/flatten

import numpy as np
from numpy import random as nr

arr =np.arange(6).reshape(2, -1)
print(arr)
# 按照列优先，展平
print("arr.ravel('F')")
print(arr.ravel('F'))
# 按照行优先，展平
print("arr.ravel()")
print(arr.ravel())

print('...................')
a =np.floor(10*np.random.random((3,4)))
print(a)
print(a.flatten())