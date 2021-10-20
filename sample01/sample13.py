##1.2 demo squeeze

import numpy as np
from numpy import random as nr

arr =np.arange(3).reshape(3, 1)
print(arr.shape)  #(3,1)
print(arr.squeeze().shape)  #(3,)
arr1 =np.arange(6).reshape(3,1,2,1)
print(arr1.shape) #(3, 1, 2, 1)
print(arr1.squeeze().shape) #(3, 2)