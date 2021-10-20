##1.2 demo resize/reshape/T

import numpy as np
from numpy import random as nr

arr =np.arange(10)
print(arr)
# 将向量 arr 维度变换为2行5列
arr.resize(2, 5)
print(arr)

print('.............................')
arr =np.arange(12).reshape(3,4)
# 向量 arr 为3行4列
print(arr)
# 将向量 arr 进行转置为4行3列
print(arr.T)