import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    """
    Inception块,四个通道px分别进行不同大小卷积核的卷积
    每个通道都使用合适的填充和步幅以输出一致的宽高（和输入一致）
    """
    # c1--c4/output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1*1 conv(path 1)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 3*3conv (path 2)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)#减小通道维数，降低模型复杂度
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) #(w-k+1+2*p)/s
        # 5*5conv(path 3)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 3*3 maximum pooling
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)#最大池化
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1=F.relu(self.p1_1(x))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4=F.relu(self.p4_2(F.relu(self.p4_1(x))))

        return torch.cat((p1,p2,p3,p4),dim=1)#4个通道 在通道维度上进行联结

class ActiveDecisionModel(nn.Module):
    """
    GoogleNet使用9个Inception块和最大池化层的（降低模型维度，复杂度）堆叠进行预测
    """
    def __init__(self,in_channels,**kwargs):
        super(ActiveDecisionModel,self).__init__(**kwargs)
        self.b1=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),#(w-7+1+6)/2 高宽减半
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size=3,stride=2,padding=1))#half of width and height
        self.b2=nn.Sequential(nn.Conv2d(64,64,kernel_size=1),
                              nn.ReLU(),
                              nn.Conv2d(64,192,kernel_size=3,padding=1),#通道数乘3倍
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.b3=nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                              Inception(256,128,(128,192),(32,96),64),
                              nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),#输出通道数是1024
                   nn.AdaptiveAvgPool2d((1,1)),#overall Avg Pool
                   nn.Flatten())#channel as output
        self.network=nn.Sequential(self.b1,self.b2,self.b3,self.b4,self.b5,
                                   nn.Linear(1024,128),
                                   nn.Linear(128,6))
    def forward(self,x):
        return self.network(x)

if __name__ == '__main__':
    model = ActiveDecisionModel(4)

    x = torch.rand(size=(16, 4, 224, 224))#2064*1544 image(RGB-D图像)

    for layer in model.network:
        x = layer(x)
        print("shape:", x.shape)

    print(x)

