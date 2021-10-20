import torch

x=torch.tensor([1,2])
y=torch.tensor([3,4])
print(x)
print(y)
z=x.add(y)
print(z)

x.add_(y)
print(x)