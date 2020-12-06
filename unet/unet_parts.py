""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

#HQC,该方法创建一个每层的卷积过程，每一层下采样或是上采样都连续进行两次两次卷积操作，这种操作将在UNet网络中重复很多次
class DoubleConv(nn.Module):
    #HQC,初始化通道数，in_channels和out_channels可以灵活设定，以便扩展使用
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            #HQC,nn.Sequential是一个时序容器，Modules 会以它们传入的顺序被添加到容器中,下面代码的操作顺序：卷积->BN->ReLU->卷积->BN->ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#HQC,下采样方法,以下代码为一个maxpool池化层，进行下采样，然后接一个DoubleConv模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    #HQC,用于反向传播
    def forward(self, x):
        return self.maxpool_conv(x)

#HQC:上采样模块，
class Up(nn.Module):
    #HQC:初始化过程，定义上采样方法以及卷积采用DoubleConv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #HQC:ConvTranspose2d为反卷积方法
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    #HQC:前向传播函数，x1接收上采样的数据，再进行concat
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#HQC:要根据分割数量，整合输出通道
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
