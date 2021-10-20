##1.2 get ndarray item by np.choice

import numpy as np
from numpy import random as nr

print('begin test 9.........................................')
a=np.arange(1,25,dtype=float)
c1=nr.choice(a,size=(3,4))  #从一个给定的1维数组中随机取样,size指定输出数组形状
c2=nr.choice(a,size=(3,4),replace=False)  #replace缺省为True，即可重复抽取。
#下式中参数p指定每个元素对应的抽取概率，缺省为每个元素被抽取的概率相同。
c3=nr.choice(a,size=(3,4),p=a / np.sum(a))
print(a)
# print("随机可重复抽取")
print(c1)
# print("随机但不重复抽取")
print(c2)
# print("随机但按制度概率抽取")
print(c3)
print('end test 9.........................................')