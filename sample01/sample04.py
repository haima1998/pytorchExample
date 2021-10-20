##1.3 gen numpy ndarray with np.arange and np.linspace

import numpy as np
from numpy import random as nr

print('begin test 7.........................................')
print(np.arange(10))
# [0 1 2 3 4 5 6 7 8 9]
print(np.arange(0, 10))
# [0 1 2 3 4 5 6 7 8 9]
print(np.arange(1, 4, 0.5))
# [1.  1.5 2.  2.5 3.  3.5]
print(np.arange(9, -1, -1))
# [9 8 7 6 5 4 3 2 1 0]

print(np.linspace(0, 1, 10))
print('end test 7.........................................')

