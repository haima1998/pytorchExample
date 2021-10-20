##1.3 gen numpy ndarray with multi dim shape
##1.4 save ndarray

import numpy as np
from numpy import random as nr

# test 5
print('begin test 5.........................................')
# 生成全是 0 的 3x3 矩阵
nd5 =np.zeros([3, 3])
#生成与nd5形状一样的全0矩阵
#np.zeros_like(nd5)
# 生成全是 1 的 3x3 矩阵
nd6 = np.ones([3, 3])
# 生成 3 阶的单位矩阵
nd7 = np.eye(3)
# 生成 3 阶对角矩阵
nd8 = np.diag([1, 2, 3])

print(nd5)
print(nd6)
print(nd7)
print(nd8)
print('end test 5.........................................')


# test 6
#np.savetxt('data/task.txt', self.task, fmt="%d", delimiter=" ")
#data/task.txt：参数为文件路径以及TXT文本名
#self.task： 为要保存的数组名
#fmt="%d"： 为指定保存的文件格式，这里为十进制
#delimiter=" "表示分隔符，这里以空格的形式隔开
print('begin test 6.........................................')
nd9 =np.random.random([5, 5])
np.savetxt(X=nd9, fname='./test1.txt')
nd10 = np.loadtxt('./test1.txt')
print(nd10)
print('end test 6.........................................')

