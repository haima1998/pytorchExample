##1.2 demo broadcast

import numpy as np
from numpy import random as nr
import time
import math

A = np.arange(0, 40,10).reshape(4, 1)
B = np.arange(0, 3)
print("A.shape:{},B.shape:{}".format(A.shape,B.shape))
print(A)
print(B)
C=A+B
print("C.shape:{}".format(C.shape))
print(C)