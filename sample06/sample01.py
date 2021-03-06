import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.bn1 = nn.BatchNorm2d(16)  # bn param = 16
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=5,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense1 = nn.Linear(900,128)
        self.dense2 = nn.Linear(128,10)
      

    def forward(self,x):
        x=self.pool1(F.relu(self.bn1(self.conv1(x)))) 
        # x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(-1,900)
        x=F.relu(self.dense2(F.relu(self.dense1(x))))
        return x

# 池化窗口为正方形 size=3, stride=2
m1 = nn.MaxPool2d(3, stride=2)

# 池化窗口为非正方形
m2 = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m2(input)
print('input shape:{} output shape:{}'.format(input.shape,output.shape))

# 输出大小为5x7
m = nn.AdaptiveMaxPool2d((5,7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
print('input shape:{} output shape:{}'.format(input.shape,output.shape))

# t输出大小为正方形 7x7 
m = nn.AdaptiveMaxPool2d(7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
print('input shape:{} output shape:{}'.format(input.shape,output.shape))

# 输出大小为 10x7
m = nn.AdaptiveMaxPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
print('input shape:{} output shape:{}'.format(input.shape,output.shape))

# 输出大小为 1x1
m = nn.AdaptiveMaxPool2d((1))
input = torch.randn(1, 64, 10, 9)
output = m(input)
print('input shape:{} output shape:{}'.format(input.shape,output.shape))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CNNNet()
net=net.to(device)

# F. show net all layers
print('...............F. show net all layers.....................')
for m in net.modules():
    print('m:{}'.format(m))
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        print('weight count:{} bias count:{}'.format(m.weight.numel(),m.bias.numel()))    


# F. show net all parameter
print('...............F. show net all parameter.....................')
for name, param in net.named_parameters():
    # if name in ['bias']:
    # print(param.size())
    print('.............................................')
    print('name:{} requires_grad:{}'.format(name,param.requires_grad))
    print('param.shape:{} total count:{}'.format(param.size(),param.numel()))
    # print('param content:{}'.format(param))