import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision import transforms, utils

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('gpu status:{}.'.format(torch.cuda.is_available()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        # conv1 weight count: 5 X 5 X 3 X 16 = 1200  bias count : 16
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # conv2 weight count: 3 X 3 X 16 X 36 = 5184  bias count : 36
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # fc1 weight count: 1296 X 128 = 165888  bias count : 128
        self.fc1 = nn.Linear(1296,128)
        # fc1 weight count: 128 X 10 = 1280  bias count : 10
        self.fc2 = nn.Linear(128,10)      

    def forward(self,x):
        #intput shape: 1, 3, 32, 32
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1,36*6*6)
        # x=F.relu(self.fc2(F.relu(self.fc1(x))))  #if rm this line,can make output onnx no fc layer
        return x

net = CNNNet()
net=net.to(device)

LR=0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


#初始化数据
for m in net.modules():
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)#卷积层参数初始化
        nn.init.constant_(m.bias, 0)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight)#全连接层参数初始化

#训练模型
# for epoch in range(2):  

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # 获取训练数据
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         # 权重参数梯度清零
#         optimizer.zero_grad()

#         # 正向及反向传播
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # 显示损失值
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

# A. save or load model
# torch.save(net, "./weight/cnn.pth")
net = torch.load("./weight/cnn.pth")
print('Finished loading pth')

# net.fc2 = Identity()
# print('model dict:{}'.format(net.state_dict()))
# print('model model:{}'.format(net.model ))
# my_net = nn.Sequential(*list(net.modules())[:-1])
print('................begin modules................')
print(list(net.modules()))
# print(list(my_net))
print('................end modules................')
# print('model:{}'.format(net))

# B. export model to onnx
torch.onnx.export(net, torch.ones((1, 3, 32, 32)).to('cpu'),
                      'weight/cnn.onnx',
                      verbose=True, opset_version=12, input_names=['images'],
                      output_names=['output'])
print('finish export onnx')

net.eval()

# C. get one test set image
writer = SummaryWriter(log_dir='logs',comment='feature map')
for i, data in enumerate(testloader, 0):
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        print('i:{} inputs {}'.format(i,inputs[0].shape))
        print('i:{} labels {}'.format(i,labels[0]))
        x=inputs[0].unsqueeze(0)
        print('i:{} x.shape:{} x.size:{}'.format(i,x.shape,x.size(0)))
        break

# D. dump input image
img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)
writer.add_image(f'ori_image', img_grid, global_step=0)
fig = plt.figure()
plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
plt.show()
utils.save_image(img_grid,'output/test02.png')

# E. show dump feature map, calc layers by layers
print('......................E. show dump feature map, calc layers by layers........................')
for name, layer in net._modules.items():
    print('............................................')
    print('name:{}'.format(name))
    print('layer:{}'.format(layer))
    print('before x.shape:{}'.format(x.shape))

    # 为fc层预处理x
    x = x.view(x.size(0), -1) if "fc" in name else x
    print(x.size())

    x = layer(x)
    print('after calc x.shape:{}'.format(x.shape))

    # 查看卷积层的特征图
    if  'layer' in name or 'conv' in name:
        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        print('dump feature map name:{} shape:{}'.format(name,x1.shape))
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
        writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
        out_image_name = "output/feature_maps_" + name + ".png"
        fig = plt.figure()
        plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
        plt.show()
        utils.save_image(img_grid,out_image_name)        
        print('write image name:{}'.format(out_image_name))

# F. show net all layers
print('...............F. show net all layers.....................')
# # conv1 weight count: 5 X 5 X 3 X 16 = 1200  bias count : 16
# self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
# self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
# # conv2 weight count: 3 X 3 X 16 X 36 = 5184  bias count : 36
# self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
# self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# # fc1 weight count: 1296 X 128 = 165888  bias count : 128
# self.fc1 = nn.Linear(1296,128)
# # fc1 weight count: 128 X 10 = 1280  bias count : 10
# self.fc2 = nn.Linear(128,10)      
for m in net.modules():
    print('m:{}'.format(m))
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        print('weight count:{} bias count:{}'.format(m.weight.numel(),m.bias.numel()))    