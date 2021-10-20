##1.2 demo np.multiply

import numpy as np
from numpy import random as nr


A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
print(A)
print(B)
print(A*B)

#或另一种表示方法
print(np.multiply(A,B))


print(A*2.0)
print(A/2.0)