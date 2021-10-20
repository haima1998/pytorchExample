import torch

a=torch.tensor([2, 3])
b=torch.tensor([3, 4])
print(a)
print(b)
c=torch.dot(a,b)  #运行结果为18
print(c)
x=torch.randint(10,(2,3))
y=torch.randint(6,(3,4))
print(x)
print(y)
e=torch.mm(x,y)
print(e)
x=torch.randint(10,(2,2,3))
y=torch.randint(6,(2,3,4))
torch.bmm(x,y)