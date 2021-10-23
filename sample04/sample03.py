###tensorboard --logdir=<your_log_dir>  see the graph

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        # input: 32 X 1 X 28 X 28
        # conv1 output: 32 X 10 X 28 X 28
        # max_pool output: 32 X 10 X 12 X 12
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        # conv2 input: 32 X 20 X 12 X 12
        # conv2 output: 32 X 20 X 8 X 8
        # max_pool output: 32 X 20 X 4 X 4
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        # x.view input: 32 X 20 X 4 X 4
        # x.view output: 32 X 320
        x = x.view(-1, 320)
        # fc1 input: 32 X 320
        # fc1 output: 32 X 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # fc2 input: 32 X 50
        # fc2 output: 32 X 10
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

#定义输入
input = torch.rand(32, 1, 28, 28)
#实例化神经网络
model = Net()
#将model保存为graph
with SummaryWriter(log_dir='logs',comment='Net') as w:
    w.add_graph(model, (input, ))


