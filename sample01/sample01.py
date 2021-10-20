##1.1 gen numpy ndarray


import numpy as np
from numpy import random as nr

# test 1
print('begin test 1.........................................')
lst1 = [3.14, 2.17, 0, 1, 2]
nd1 =np.array(lst1)
print(nd1)
print(type(nd1))
print(type(lst1))
print('end test 1.........................................')

# test 2
print('begin test 2.........................................')
lst2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd2 =np.array(lst2)
print(nd2)
print(type(nd2))
print('end test 2.........................................')

