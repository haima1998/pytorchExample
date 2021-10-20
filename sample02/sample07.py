import torch

t = torch.randn(1, 3)
t1 = torch.randn(3, 1)
t2 = torch.randn(1, 3)
print(t)
print(t1)
print(t2)
#t+0.1*(t1/t2)
#print(torch.addcdiv(t, 0.1, t1, t2))
#计算sigmoid
print(torch.sigmoid(t))
#将t限制在[0,1]之间
print(torch.clamp(t,0,1))
#t+2进行就地运算
print(t.add_(2))
