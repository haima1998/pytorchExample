##1.2 demo reshpae

import numpy as np
from numpy import random as nr

arr =np.arange(10)
print(arr)
# 将向量 arr 维度变换为2行5列
print(arr.reshape(2, 5))
# 指定维度时可以只指定行数或列数, 其他用 -1 代替
print(arr.reshape(5, -1))
print(arr.reshape(-1, 5))