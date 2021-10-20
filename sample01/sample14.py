##1.2 demo transpose

import numpy as np
from numpy import random as nr

arr2 = np.arange(24).reshape(2,3,4)
print(arr2.shape)  #(2, 3, 4)
print(arr2)
print(arr2.transpose(1,2,0).shape)  #(3, 4, 2)
print(arr2.transpose(1,2,0))