##1.2 demo sin

import numpy as np
from numpy import random as nr
import time
import math


x = [i * 0.001 for i in np.arange(10)]
print(x)
#start = time.clock()
for i, t in enumerate(x):
    print(i)
    print(t)
    x[i] = math.sin(t)
    print(x[i])

print('...............................')
#print ("math.sin:", time.clock() - start )

x = [i * 0.001 for i in np.arange(10)]
print(x)
x = np.array(x)
print(x)
#start = time.clock()
print(np.sin(x))

#print ("numpy.sin:", time.clock() - start )