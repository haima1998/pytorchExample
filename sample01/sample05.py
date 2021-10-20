##1.2 get ndarray item

import numpy as np
from numpy import random as nr


print('begin test 8.........................................')
np.random.seed(2019)
nd11 = np.random.random([10])
#获取指定位置的数据，获取第4个元素
print(nd11)
nd11[3]
print(nd11[3])
#截取一段数据
print(nd11[3:6])
#截取固定间隔数据
print(nd11[1:6:2])
#倒序取数
print(nd11[::-2])
#截取一个多维数组的一个区域内数据
nd12=np.arange(25).reshape([5,5])
print(nd12)
print(nd12[1:3,1:3])

#截取一个多维数组中，数值在一个值域之内的数据
print(nd12[(nd12>3)&(nd12<10)])

#截取多维数组中，指定的行,如读取第2,3行
print(nd12[[1,2]])  #或nd12[1:3,:]
##截取多维数组中，指定的列,如读取第2,3列
print(nd12[:,1:3])
print('end test 8.........................................')


