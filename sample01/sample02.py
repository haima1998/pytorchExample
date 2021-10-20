##1.2 gen numpy ndarray by np.random

import numpy as np
from numpy import random as nr

# test 3
print('begin test 3.........................................')
nd3 =np.random.random([3, 3])
print(nd3)
print(type(nd3))
print("nd3 shape:",nd3.shape)
print('end test 3.........................................')

# test 4
print('begin test 4.........................................')
np.random.seed(123)
nd4 = np.random.randn(2,3)
print(nd4)
np.random.shuffle(nd4)
print("after shuffle:")
print(nd4)
print(type(nd4))
print('end test 4.........................................')


