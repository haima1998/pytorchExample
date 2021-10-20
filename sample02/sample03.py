import torch

#根据list数据生成tensor
a=torch.Tensor([1,2,3,4,5,6])
print(a)
#根据指定形状生成tensor
b=torch.Tensor(2,3)
print(b)
#根据给定的tensor的形状
t=torch.Tensor([[1,2,3],[4,5,6]])
print(t)
#查看tensor的形状
print(t.size())
#shape与size()等价方式
print(t.shape)
#根据已有形状创建tensor

print(torch.Tensor(t.size()))

print('....................................')
t1=torch.Tensor(2)
t2=torch.tensor(2)
print("t1 value:{},torch.Tensor.type:{}".format(t1,t1.type()))
print("t2 value:{},torch.tensor.type:{}".format(t2,t2.type()))

print('....................................')
#生成一个单位矩阵
print(torch.eye(2,2))
#自动生成全是0的矩阵
print(torch.zeros(2,3))
#根据规则生成数据
print(torch.linspace(1,10,4))
#生成满足均匀分布随机数
print(torch.rand(2,3))
#生成满足标准分布随机数
print(torch.randn(2,3))
#返回所给数据形状相同，值全为0的张量
print(torch.zeros_like(torch.rand(2,3)))
