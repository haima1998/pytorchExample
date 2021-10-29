import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('trainset:{}'.format(trainset))  # 50000 images
print('testset:{}'.format(testset))    # 10000 images
print('trainloader len:{}'.format(trainloader.__len__()))  # 50000 / 4 = 12500 batch (batch size = 4)
print('testloader: len:{}'.format(testloader.__len__()))   # 10000 / 4 = 2500  batch (batch size = 4)

import matplotlib.pyplot as plt
import numpy as np

# 显示图像

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取部分训练数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

print('images:{} labels:{}'.format(images.shape,labels.shape))
# images:torch.Size([4, 3, 32, 32]) labels:torch.Size([4])

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296,128)
        self.fc2 = nn.Linear(128,10)      

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1,36*6*6)
        x=F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

class CNNNetAAP(nn.Module):
    def __init__(self):
        super(CNNNetAAP, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.pool2 = nn.MaxPool2d(2, 2)
        #使用全局平均池化层
        self.aap=nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.aap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)
        return x
#case1: FC layer
net = CNNNet()

# case2: AAP
# net = CNNNetAAP()

net=net.to(device)

print("net have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))


import torch.optim as optim
LR=0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=LR)

print('before net:{}'.format(net))
#取模型中的前四层
nn.Sequential(*list(net.children())[:4])
print('after net:{}'.format(net))


for epoch in range(10):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 权重参数梯度清零
        optimizer.zero_grad()

        # 正向及反向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 显示损失值
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#case1: FC layer
# [10,  6000] loss: 0.378
# [10,  8000] loss: 0.409
# [10, 10000] loss: 0.422
# [10, 12000] loss: 0.437

# case2: AAP
# [10,  2000] loss: 0.988
# [10,  4000] loss: 0.978
# [10,  6000] loss: 1.007
# [10,  8000] loss: 0.991
# [10, 10000] loss: 0.997
# [10, 12000] loss: 0.974

# A. save or load model
torch.save(net, "./weight/cnn.pth")
print('Finished save pth')
# net = torch.load("./weight/cnn_with_fc.pth")
# print('Finished loading pth')

dataiter = iter(testloader)
images, labels = dataiter.next()
#images, labels = images.to(device), labels.to(device)
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# GroundTruth:    cat  ship  ship plane

images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))
#case1: FC layer
# Predicted:    dog  ship   car plane

# case2: AAP
# Predicted:    cat  ship  ship plane

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# case 1: use fc layer
# Accuracy of the network on the 10000 test images: 68 %

# case2: AAP
# Accuracy of the network on the 10000 test images: 64 %

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# case 1: use fc layer
# Accuracy of plane : 62 %
# Accuracy of   car : 81 %
# Accuracy of  bird : 65 %
# Accuracy of   cat : 46 %
# Accuracy of  deer : 61 %
# Accuracy of   dog : 53 %
# Accuracy of  frog : 80 %
# Accuracy of horse : 73 %
# Accuracy of  ship : 80 %
# Accuracy of truck : 77 %

# case2: AAP
# Accuracy of plane : 56 %
# Accuracy of   car : 68 %
# Accuracy of  bird : 54 %
# Accuracy of   cat : 44 %
# Accuracy of  deer : 61 %
# Accuracy of   dog : 58 %
# Accuracy of  frog : 65 %
# Accuracy of horse : 70 %
# Accuracy of  ship : 85 %
# Accuracy of truck : 80 %


import collections
import torch

def paras_summary(input_size, model):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = collections.OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [torch.rand(1,*in_size) for in_size in input_size]
        else:
            x = torch.rand(1,*input_size)

        # create properties
        summary = collections.OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        return summary


net = CNNNet()
input_size=[3,32,32]
print('begin paras_summary')
paras_summary(input_size,net)
print('end paras_summary')

# OrderedDict([('Conv2d-1',
#               OrderedDict([('input_shape', [-1, 3, 32, 32]),
#                            ('output_shape', [-1, 16, 28, 28]),
#                            ('trainable', True),
#                            ('nb_params', tensor(1216))])),
#              ('MaxPool2d-2',
#               OrderedDict([('input_shape', [-1, 16, 28, 28]),
#                            ('output_shape', [-1, 16, 14, 14]),
#                            ('nb_params', 0)])),
#              ('Conv2d-3',
#               OrderedDict([('input_shape', [-1, 16, 14, 14]),
#                            ('output_shape', [-1, 36, 12, 12]),
#                            ('trainable', True),
#                            ('nb_params', tensor(5220))])),
#              ('MaxPool2d-4',
#               OrderedDict([('input_shape', [-1, 36, 12, 12]),
#                            ('output_shape', [-1, 36, 6, 6]),
#                            ('nb_params', 0)])),
#              ('Linear-5',
#               OrderedDict([('input_shape', [-1, 1296]),
#                            ('output_shape', [-1, 128]),
#                            ('trainable', True),
#                            ('nb_params', tensor(166016))])),
#              ('Linear-6',
#               OrderedDict([('input_shape', [-1, 128]),
#                            ('output_shape', [-1, 10]),
#                            ('trainable', True),
#                            ('nb_params', tensor(1290))]))])