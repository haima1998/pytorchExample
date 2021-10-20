import torch

#生成一个形状为2x3的矩阵
x = torch.randn(2, 3)
print(x)
#查看矩阵的形状
print(x.size())   #结果为torch.Size([2, 3])

#查看x的维度
print(x.dim())    #结果为2
#把x变为3x2的矩阵
print(x.view(3,2))
#把x展平为1维向量
y=x.view(-1)  
print(y)
print(y.shape)
#添加一个维度
z=torch.unsqueeze(y,0)
print(z)
#查看z的形状
print(z.size())   #结果为torch.Size([1, 6])
#计算Z的元素个数
print(z.numel())   #结果为6

x1=torch.arange(12).view(4,3)
x1=x1.float()
print('x1:{}'.format(x1))
x_mean0 = torch.mean(x1,dim=0)
print('x_mean0:{}'.format(x_mean0))
x_mean1 = torch.mean(x1,dim=1)
print('x_mean1:{}'.format(x_mean1))