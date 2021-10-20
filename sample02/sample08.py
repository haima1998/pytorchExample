import torch
import numpy as np

k = np.linspace(0,10,6)
print(k)

#生成一个含6个数的向量
a=torch.linspace(0,10,6)
print(a)
#使用view方法，把a变为2x3矩阵
a=a.view((2,3))
print(a)
#沿y轴方向累加，即dim=0
b=a.sum(dim=0)   #b的形状为[3]
print(b)
#沿y轴方向累加，即dim=0,并保留含1的维度
b=a.sum(dim=0,keepdim=True) #b的形状为[1,3]
print(b)


print('...............................')
x=torch.linspace(0,10,6).view(2,3)
print(x)
#求所有元素的最大值
print(torch.max(x))   #结果为10
#求y轴方向的最大值
f=torch.max(x,dim=0)  #结果为[6,8,10]
print(f)
#求最大的2个元素
d=torch.topk(x,1,dim=0)  #结果为[6,8,10],对应索引为tensor([[1, 1, 1]
print(d)